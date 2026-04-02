from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

from pydantic_harness.secret_masking import (
    _ALL_BUILTIN_PATTERNS,
    _BUILTIN_CATEGORIES,
    SecretMasking,
    _mask_text,
)

# --- Unit tests for _mask_text ---


class TestMaskText:
    def test_no_match_returns_original(self):
        assert _mask_text('hello world', _ALL_BUILTIN_PATTERNS, '[REDACTED]') == 'hello world'

    def test_openai_key(self):
        text = 'key is sk-abc123def456ghi789jkl012mno'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'sk-abc123' not in result
        assert '[REDACTED]' in result

    def test_anthropic_key(self):
        text = 'sk-ant-api03-abcdefghijklmnopqrstuvwxyz'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'sk-ant-' not in result
        assert result == '[REDACTED]'

    def test_aws_access_key(self):
        text = 'AWS key: AKIAIOSFODNN7EXAMPLE'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'AKIA' not in result
        assert '[REDACTED]' in result

    def test_github_token(self):
        text = 'token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'ghp_' not in result

    def test_slack_token(self):
        text = 'xoxb-123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'xoxb-' not in result

    def test_google_api_key(self):
        text = 'AIzaSyD-abcdefghijklmnopqrstuvwxyz01234'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'AIza' not in result

    def test_generic_api_key(self):
        text = 'api_key = "abcdef1234567890abcdef"'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'abcdef1234567890' not in result

    def test_bearer_token(self):
        text = 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'Bearer eyJ' not in result

    def test_jwt(self):
        text = 'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123def456'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'eyJ' not in result

    def test_password_in_url(self):
        text = 'postgresql://admin:s3cret_pass@db.example.com:5432/mydb'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 's3cret_pass' not in result

    def test_database_connection_string(self):
        text = 'mongodb+srv://user:pass@cluster.mongodb.net/db'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'user:pass' not in result

    def test_private_key(self):
        text = '-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAK...'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert '-----BEGIN RSA PRIVATE KEY-----' not in result

    def test_ec_private_key(self):
        text = '-----BEGIN EC PRIVATE KEY-----\ndata...'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert '-----BEGIN EC PRIVATE KEY-----' not in result

    def test_openssh_private_key(self):
        text = '-----BEGIN OPENSSH PRIVATE KEY-----\ndata...'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert '-----BEGIN OPENSSH PRIVATE KEY-----' not in result

    def test_multiple_secrets_in_one_string(self):
        text = 'key=sk-abc123def456ghi789jkl012mno, token=ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '[REDACTED]')
        assert 'sk-abc123' not in result
        assert 'ghp_' not in result
        assert result.count('[REDACTED]') == 2

    def test_custom_replacement(self):
        text = 'sk-abc123def456ghi789jkl012mno'
        result = _mask_text(text, _ALL_BUILTIN_PATTERNS, '***')
        assert result == '***'


# --- Tests for SecretMasking dataclass construction ---


class TestSecretMaskingInit:
    def test_defaults(self):
        sm = SecretMasking()
        assert sm.categories is None
        assert sm.custom_patterns is None
        assert sm.replacement == '[REDACTED]'
        assert sm._compiled == _ALL_BUILTIN_PATTERNS

    def test_specific_categories(self):
        sm = SecretMasking(categories=['api_keys', 'tokens'])
        expected = {**_BUILTIN_CATEGORIES['api_keys'], **_BUILTIN_CATEGORIES['tokens']}
        assert sm._compiled == expected

    def test_single_category(self):
        sm = SecretMasking(categories=['private_keys'])
        assert sm._compiled == _BUILTIN_CATEGORIES['private_keys']

    def test_unknown_category_raises(self):
        with pytest.raises(ValueError, match="Unknown secret pattern category 'bogus'"):
            SecretMasking(categories=['bogus'])

    def test_custom_patterns(self):
        sm = SecretMasking(custom_patterns={'my_secret': r'SECRET-\d{6}'})
        assert 'my_secret' in sm._compiled
        assert sm._compiled['my_secret'].pattern == r'SECRET-\d{6}'

    def test_custom_patterns_with_categories(self):
        sm = SecretMasking(categories=['api_keys'], custom_patterns={'my_secret': r'SECRET-\d{6}'})
        assert 'openai_key' in sm._compiled
        assert 'my_secret' in sm._compiled
        assert 'bearer_token' not in sm._compiled

    def test_custom_replacement(self):
        sm = SecretMasking(replacement='<MASKED>')
        assert sm.replacement == '<MASKED>'


# --- Tests for after_tool_execute ---


class TestAfterToolExecute:
    @pytest.fixture()
    def capability(self) -> SecretMasking:
        return SecretMasking()

    @pytest.fixture()
    def ctx(self) -> Any:
        return MagicMock()

    @pytest.fixture()
    def call(self) -> Any:
        return MagicMock()

    @pytest.fixture()
    def tool_def(self) -> Any:
        return MagicMock()

    @pytest.mark.anyio()
    async def test_string_result_with_secret(self, capability: SecretMasking, ctx: Any, call: Any, tool_def: Any):
        result = await capability.after_tool_execute(
            ctx, call=call, tool_def=tool_def, args={}, result='key: sk-abc123def456ghi789jkl012mno'
        )
        assert isinstance(result, str)
        assert 'sk-abc123' not in result
        assert '[REDACTED]' in result

    @pytest.mark.anyio()
    async def test_string_result_without_secret(self, capability: SecretMasking, ctx: Any, call: Any, tool_def: Any):
        result = await capability.after_tool_execute(ctx, call=call, tool_def=tool_def, args={}, result='hello world')
        assert result == 'hello world'

    @pytest.mark.anyio()
    async def test_non_string_result_with_secret(self, capability: SecretMasking, ctx: Any, call: Any, tool_def: Any):
        result = await capability.after_tool_execute(
            ctx, call=call, tool_def=tool_def, args={}, result=['key', 'sk-abc123def456ghi789jkl012mno']
        )
        assert isinstance(result, str)
        assert 'sk-abc123' not in result

    @pytest.mark.anyio()
    async def test_non_string_result_without_secret(
        self, capability: SecretMasking, ctx: Any, call: Any, tool_def: Any
    ):
        result = await capability.after_tool_execute(
            ctx, call=call, tool_def=tool_def, args={}, result={'status': 'ok'}
        )
        assert result == {'status': 'ok'}

    @pytest.mark.anyio()
    async def test_custom_replacement(self, ctx: Any, call: Any, tool_def: Any):
        capability = SecretMasking(replacement='<HIDDEN>')
        result = await capability.after_tool_execute(
            ctx, call=call, tool_def=tool_def, args={}, result='sk-abc123def456ghi789jkl012mno'
        )
        assert result == '<HIDDEN>'

    @pytest.mark.anyio()
    async def test_custom_pattern(self, ctx: Any, call: Any, tool_def: Any):
        capability = SecretMasking(categories=[], custom_patterns={'internal': r'INT-[A-Z]{8}'})
        result = await capability.after_tool_execute(
            ctx, call=call, tool_def=tool_def, args={}, result='secret: INT-ABCDEFGH'
        )
        assert 'INT-ABCDEFGH' not in result
        assert '[REDACTED]' in result


# --- Tests for after_model_request ---


class TestAfterModelRequest:
    @pytest.fixture()
    def capability(self) -> SecretMasking:
        return SecretMasking()

    @pytest.fixture()
    def ctx(self) -> Any:
        return MagicMock()

    @pytest.fixture()
    def request_context(self) -> Any:
        return MagicMock()

    def _make_response(self, *texts: str) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=t) for t in texts])

    @pytest.mark.anyio()
    async def test_scrubs_text_parts(self, capability: SecretMasking, ctx: Any, request_context: Any):
        response = self._make_response('Your key is sk-abc123def456ghi789jkl012mno')
        result = await capability.after_model_request(ctx, request_context=request_context, response=response)
        assert isinstance(result.parts[0], TextPart)
        assert 'sk-abc123' not in result.parts[0].content
        assert '[REDACTED]' in result.parts[0].content

    @pytest.mark.anyio()
    async def test_clean_text_unchanged(self, capability: SecretMasking, ctx: Any, request_context: Any):
        response = self._make_response('No secrets here')
        result = await capability.after_model_request(ctx, request_context=request_context, response=response)
        assert isinstance(result.parts[0], TextPart)
        assert result.parts[0].content == 'No secrets here'

    @pytest.mark.anyio()
    async def test_multiple_parts(self, capability: SecretMasking, ctx: Any, request_context: Any):
        response = self._make_response(
            'key: AKIAIOSFODNN7EXAMPLE',
            'clean text',
            'token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn',
        )
        result = await capability.after_model_request(ctx, request_context=request_context, response=response)
        parts = result.parts
        assert isinstance(parts[0], TextPart)
        assert 'AKIA' not in parts[0].content
        assert isinstance(parts[1], TextPart)
        assert parts[1].content == 'clean text'
        assert isinstance(parts[2], TextPart)
        assert 'ghp_' not in parts[2].content

    @pytest.mark.anyio()
    async def test_non_text_parts_are_untouched(self, capability: SecretMasking, ctx: Any, request_context: Any):
        tool_call = ToolCallPart(tool_name='get_secret', args='{}')
        response = ModelResponse(parts=[tool_call])
        result = await capability.after_model_request(ctx, request_context=request_context, response=response)
        assert result.parts[0] is tool_call


# --- Test pattern categories ---


class TestPatternCategories:
    def test_all_categories_exist(self):
        assert set(_BUILTIN_CATEGORIES) == {'api_keys', 'tokens', 'connection_strings', 'private_keys'}

    def test_api_keys_category(self):
        patterns = _BUILTIN_CATEGORIES['api_keys']
        assert 'openai_key' in patterns
        assert 'anthropic_key' in patterns
        assert 'aws_access_key' in patterns
        assert 'github_token' in patterns
        assert 'slack_token' in patterns
        assert 'google_api_key' in patterns
        assert 'generic_api_key' in patterns

    def test_tokens_category(self):
        patterns = _BUILTIN_CATEGORIES['tokens']
        assert 'bearer_token' in patterns
        assert 'jwt' in patterns

    def test_connection_strings_category(self):
        patterns = _BUILTIN_CATEGORIES['connection_strings']
        assert 'password_in_url' in patterns
        assert 'database_connection' in patterns

    def test_private_keys_category(self):
        patterns = _BUILTIN_CATEGORIES['private_keys']
        assert 'private_key' in patterns

    def test_all_builtin_is_union_of_categories(self):
        expected: dict[str, re.Pattern[str]] = {}
        for cat_patterns in _BUILTIN_CATEGORIES.values():
            expected.update(cat_patterns)
        assert _ALL_BUILTIN_PATTERNS == expected


# --- Edge cases ---


class TestEdgeCases:
    def test_empty_categories_list_with_custom(self):
        sm = SecretMasking(categories=[], custom_patterns={'test': r'TEST-\d+'})
        # Only custom patterns, no builtins.
        assert 'test' in sm._compiled
        assert 'openai_key' not in sm._compiled

    def test_empty_categories_no_custom(self):
        sm = SecretMasking(categories=[])
        assert sm._compiled == {}

    @pytest.mark.anyio()
    async def test_empty_string_tool_result(self):
        sm = SecretMasking()
        ctx = MagicMock()
        result = await sm.after_tool_execute(ctx, call=MagicMock(), tool_def=MagicMock(), args={}, result='')
        assert result == ''

    @pytest.mark.anyio()
    async def test_none_tool_result(self):
        sm = SecretMasking()
        ctx = MagicMock()
        result = await sm.after_tool_execute(ctx, call=MagicMock(), tool_def=MagicMock(), args={}, result=None)
        assert result is None
