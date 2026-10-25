"""Offline protocol tests: SDK imports and model responses are stubbed, never sent."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def load_script(name):
    path = Path(__file__).with_name(name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        'openai': SimpleNamespace(OpenAI=Mock()),
        'dotenv': SimpleNamespace(load_dotenv=lambda: None),
    }):
        spec.loader.exec_module(module)
    return module


v1 = load_script('agent_workflow_demo')
v2 = load_script('agent_workflow_v2')


class Message:
    def __init__(self, content=None, calls=()):
        self.content = content
        self.tool_calls = [SimpleNamespace(
            id=call_id, type='function',
            function=SimpleNamespace(name=name, arguments=args),
        ) for call_id, name, args in calls]

    def model_dump(self, exclude_none=True):
        result = {'role': 'assistant'}
        if self.content is not None:
            result['content'] = self.content
        if self.tool_calls:
            result['tool_calls'] = [
                {'id': tc.id, 'type': tc.type, 'function': {
                    'name': tc.function.name, 'arguments': tc.function.arguments,
                }} for tc in self.tool_calls
            ]
        return result


class FakeCompletions:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        pending = []
        for message in kwargs['messages']:
            if message['role'] == 'assistant':
                pending.extend(tc['id'] for tc in message.get('tool_calls', []))
            elif message['role'] == 'tool':
                assert message['tool_call_id'] in pending, 'orphan tool result'
                pending.remove(message['tool_call_id'])
        assert not pending, 'missing tool result'
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(choices=[SimpleNamespace(message=response)])


def agent_with(*responses, cls=v2.MultimodalAgentV2):
    agent = cls.__new__(cls)
    completions = FakeCompletions(*responses)
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    agent.model_id = 'offline-test'
    return agent, completions


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.image = Path(self.temp.name) / 'sample.png'
        self.image.write_bytes(b'offline fixture; not decoded by a model')

    def test_multiple_tools_keep_image_and_assistant_calls(self):
        calls = [('lookup', 'knowledge_lookup', '{"error_keywords":"E42"}'),
                 ('extract', 'structured_extract', '{"fields":["amount"]}')]
        agent, api = agent_with(Message(None, calls), Message('Tools are not implemented.'))
        result = agent.run(str(self.image), 'Explain')
        self.assertEqual(result['final_answer'], 'Tools are not implemented.')
        messages = api.requests[1]['messages']
        self.assertEqual(messages[:2], api.requests[0]['messages'])
        self.assertEqual(messages[2], Message(None, calls).model_dump())
        self.assertEqual([m['tool_call_id'] for m in messages[3:]], ['lookup', 'extract'])
        self.assertNotIn('messages', result)
        for out in result['tool_outputs']:
            self.assertEqual(json.loads(out['output'])['status'], 'not_implemented')

    def test_no_tools_returns_original_answer_without_second_call(self):
        agent, api = agent_with(Message('Direct answer'))
        self.assertEqual(agent.run(str(self.image), 'Explain')['final_answer'], 'Direct answer')
        self.assertEqual(len(api.requests), 1)

    def test_empty_question_makes_no_request(self):
        agent, api = agent_with()
        self.assertEqual(agent.run(str(self.image), '  ')['status'], 'error')
        self.assertEqual(api.requests, [])

    def test_bad_arguments_return_errors_and_allow_other_tools(self):
        bad = ['{', '[]', 'null', '{}', '{"fields":"amount"}',
               '{"fields":[1]}', '{"fields":[]}', '{"fields":[" "]}']
        for args in bad:
            with self.subTest(args=args):
                calls = [('bad', 'structured_extract', args),
                         ('good', 'knowledge_lookup', '{"error_keywords":"E42"}')]
                agent, api = agent_with(Message(None, calls), Message('Could not extract.'))
                result = agent.run(str(self.image), 'Explain')
                self.assertEqual(json.loads(result['tool_outputs'][0]['output'])['status'], 'error')
                self.assertEqual(json.loads(result['tool_outputs'][1]['output'])['status'], 'not_implemented')
                self.assertEqual(api.requests[1]['messages'][2]['tool_calls'][0]['function']['arguments'], args)

    def test_invalid_lookup_arguments(self):
        agent, _ = agent_with()
        for args in ('{}', '{"error_keywords":2}', '{"error_keywords":" "}'):
            out = agent.execute_tools([{'id': 'x', 'name': 'knowledge_lookup', 'arguments': args}])
            self.assertEqual(json.loads(out[0]['output'])['status'], 'error')

    def test_unknown_tool_is_not_dispatched(self):
        agent, _ = agent_with(Message(None, [('x', '__getattribute__', '{}')]), Message('Unknown tool.'))
        out = agent.run(str(self.image), 'Explain')['tool_outputs'][0]
        self.assertIn('Unknown tool', json.loads(out['output'])['error'])

    def test_tool_exception_becomes_paired_error(self):
        agent, _ = agent_with(Message(None, [('x', 'knowledge_lookup', '{"error_keywords":"E42"}')]), Message('Failed.'))
        agent.knowledge_lookup = Mock(side_effect=RuntimeError('offline failure'))
        out = agent.run(str(self.image), 'Explain')['tool_outputs'][0]
        self.assertEqual(json.loads(out['output'])['error'], 'offline failure')

    def test_placeholders_never_fabricate_values(self):
        agent, _ = agent_with()
        out = json.loads(agent.structured_extract(['amount', 'date', 'merchant']))
        self.assertEqual(out['values'], {'amount': None, 'date': None, 'merchant': None})
        self.assertEqual(out['status'], 'not_implemented')
        self.assertEqual(json.loads(agent.knowledge_lookup('E42'))['status'], 'not_implemented')

    def test_missing_image_makes_no_request(self):
        agent, api = agent_with()
        with self.assertRaises(FileNotFoundError):
            agent.run(str(self.image) + '.missing', 'Explain')
        self.assertEqual(api.requests, [])

    def test_empty_model_answer_fails_explicitly(self):
        for content in (None, '', ' '):
            with self.subTest(content=content):
                agent, _ = agent_with(Message(content))
                with self.assertRaisesRegex(ValueError, 'no answer'):
                    agent.run(str(self.image), 'Explain')

    def test_empty_final_answer_fails_explicitly(self):
        agent, _ = agent_with(Message(None, [('x', 'knowledge_lookup', '{"error_keywords":"E42"}')]), Message(None))
        with self.assertRaisesRegex(ValueError, 'no final answer'):
            agent.run(str(self.image), 'Explain')

    def test_main_reports_path_error_as_json(self):
        for module in (v1, v2):
            with self.subTest(module=module.__name__):
                stream = io.StringIO()
                with patch.object(module, 'resolve_image_path', side_effect=FileNotFoundError('missing image')), contextlib.redirect_stdout(stream):
                    self.assertEqual(module.main(), 1)
                self.assertEqual(json.loads(stream.getvalue())['status'], 'error')

    def test_main_reports_api_error_as_json(self):
        for module, cls in ((v1, v1.MultimodalAgent), (v2, v2.MultimodalAgentV2)):
            agent, _ = agent_with(RuntimeError('offline API failure'), cls=cls)
            stream = io.StringIO()
            with patch.object(module, 'resolve_image_path', return_value=self.image), patch.object(module, cls.__name__, return_value=agent), contextlib.redirect_stdout(stream):
                self.assertEqual(module.main(), 1)
            self.assertEqual(json.loads(stream.getvalue())['error'], 'offline API failure')

    def test_main_success_and_empty_question_exit_codes(self):
        for module, cls in ((v1, v1.MultimodalAgent), (v2, v2.MultimodalAgentV2)):
            for question, expected in (('Explain', 0), (' ', 1)):
                agent, _ = agent_with(Message('Answer'), cls=cls)
                with patch.dict('os.environ', {'QUESTION': question}), patch.object(module, 'resolve_image_path', return_value=self.image), patch.object(module, cls.__name__, return_value=agent), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(module.main(), expected)

    def test_v1_rejects_missing_perception(self):
        agent, _ = agent_with(Message(None), cls=v1.MultimodalAgent)
        with self.assertRaisesRegex(ValueError, 'no perception summary'):
            agent.run(str(self.image), 'Explain')


if __name__ == '__main__':
    unittest.main()
