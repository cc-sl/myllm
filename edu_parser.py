# edu_parser.py
"""
EduAssist 递归下降语法分析器
文法定义（LL(1)）：
  Feedback       → FEEDBACK LBRACE FieldList RBRACE
  FieldList      → Field FieldList | ε
  Field          → ScoreField | LevelField | CommentBlock | ErrorList
  ScoreField     → SCORE COLON NUMBER SEMICOLON
  LevelField     → LEVEL COLON IDENT SEMICOLON
  CommentBlock   → COMMENT LBRACE CommentFieldList RBRACE
  CommentFieldList → CommentField CommentFieldList | ε
  CommentField   → TextField | SuggestionField
  TextField      → TEXT COLON STRING SEMICOLON
  SuggestionField → SUGGESTION COLON STRING SEMICOLON
  ErrorList      → ERRORS LBRACKET ErrorItems RBRACKET
  ErrorItems     → ErrorItem ErrorItems | ε
  ErrorItem      → ERROR LPAREN ErrorParams RPAREN SEMICOLON
  ErrorParams    → ErrorParam RestParams
  RestParams     → COMMA ErrorParam RestParams | ε
  ErrorParam     → IDENT COLON ErrorValue
  ErrorValue     → NUMBER | STRING | IDENT
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Tuple


# ============================================================
# AST 节点定义
# ============================================================
@dataclass
class ASTNode:
    """抽象语法树节点"""
    node_type: str
    value: Any = None
    children: List['ASTNode'] = field(default_factory=list)

    def pprint(self, indent: int = 0) -> None:
        """以树形结构打印 AST"""
        prefix = "  " * indent
        if self.value is not None and self.children:
            print(f"{prefix}{self.node_type}({self.value})")
        elif self.value is not None:
            print(f"{prefix}{self.node_type}({self.value})")
        else:
            print(f"{prefix}{self.node_type}")
        for child in self.children:
            child.pprint(indent + 1)

    def to_dict(self) -> dict:
        """将 AST 转换为字典（便于 JSON 序列化）"""
        result = {"type": self.node_type}
        if self.value is not None:
            result["value"] = self.value
        if self.children:
            result["children"] = [c.to_dict() for c in self.children]
        return result


# ============================================================
# 自定义异常
# ============================================================
class ParseError(Exception):
    """语法分析错误，包含行号和期望的 Token 类型"""

    def __init__(self, message: str, line: int = None,
                 col: int = None, expected: str = None, got: str = None):
        self.line = line
        self.col = col
        self.expected = expected
        self.got = got
        detail = message
        if line is not None:
            detail += f" (line {line}"
            if col is not None:
                detail += f", col {col}"
            detail += ")"
        if expected is not None:
            detail += f" [expected: {expected}, got: {got}]"
        super().__init__(detail)


# ============================================================
# 词法分析器（Lexer）
# ============================================================
class Lexer:
    """
    简易词法分析器，将 EduAssist 反馈文本转换为 Token 列表。
    Token 格式: (type, value, line, col)
    """

    KEYWORDS = {
        'FEEDBACK', 'SCORE', 'LEVEL', 'COMMENT',
        'TEXT', 'SUGGESTION', 'ERRORS', 'ERROR'
    }

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.line = 1
        self.col = 1

    def _advance(self, ch: str = None) -> None:
        """推进一个字符位置"""
        if ch is None and self.pos < len(self.text):
            ch = self.text[self.pos]
        self.pos += 1
        if ch == '\n':
            self.line += 1
            self.col = 1
        else:
            self.col += 1

    def _peek(self) -> str:
        """查看当前字符"""
        if self.pos < len(self.text):
            return self.text[self.pos]
        return '\0'

    def tokenize(self) -> List[Tuple[str, Any, int, int]]:
        """将文本转换为 Token 列表"""
        tokens = []
        while self.pos < len(self.text):
            ch = self._peek()

            # 跳过空白
            if ch in ' \t\r\n':
                self._advance(ch)
                continue

            # 跳过单行注释 //
            if ch == '/' and self.pos + 1 < len(self.text) and self.text[self.pos + 1] == '/':
                while self.pos < len(self.text) and self.text[self.pos] != '\n':
                    self._advance(self.text[self.pos])
                continue

            start_line, start_col = self.line, self.col

            # 标点符号
            punct_map = {
                '{': 'LBRACE', '}': 'RBRACE',
                '[': 'LBRACKET', ']': 'RBRACKET',
                '(': 'LPAREN', ')': 'RPAREN',
                ':': 'COLON', ';': 'SEMICOLON', ',': 'COMMA'
            }
            if ch in punct_map:
                self._advance(ch)
                tokens.append((punct_map[ch], ch, start_line, start_col))
                continue

            # 字符串字面量（双引号）
            if ch == '"':
                self._advance(ch)
                string_val = []
                while self.pos < len(self.text) and self._peek() != '"':
                    c = self._peek()
                    if c == '\\':  # 转义字符
                        self._advance(c)
                        if self.pos < len(self.text):
                            string_val.append(self._peek())
                            self._advance(self.text[self.pos])
                    else:
                        string_val.append(c)
                        self._advance(c)
                if self.pos < len(self.text):
                    self._advance('"')  # 消耗右引号
                tokens.append(('STRING', ''.join(string_val), start_line, start_col))
                continue

            # 数字
            if ch.isdigit():
                num_str = []
                while self.pos < len(self.text) and self._peek().isdigit():
                    num_str.append(self._peek())
                    self._advance(self._peek())
                tokens.append(('NUMBER', int(''.join(num_str)), start_line, start_col))
                continue

            # 标识符 / 关键字
            if ch.isalpha() or ch == '_':
                ident = []
                while self.pos < len(self.text) and (self._peek().isalnum() or self._peek() == '_'):
                    ident.append(self._peek())
                    self._advance(self._peek())
                word = ''.join(ident)
                if word in self.KEYWORDS:
                    tokens.append(('KEYWORD', word, start_line, start_col))
                else:
                    tokens.append(('IDENT', word, start_line, start_col))
                continue

            # 不可识别字符
            raise ParseError(
                f"Unrecognized character: '{ch}'",
                line=self.line, col=self.col
            )

        # 文件结束标记
        tokens.append(('EOF', None, self.line, self.col))
        return tokens


# ============================================================
# 递归下降语法分析器
# ============================================================
class EduParser:
    """
    EduAssist 反馈语言的递归下降分析器。
    输入: Token 列表 (来自 Lexer)
    输出: ASTNode (抽象语法树根节点)
    """

    def __init__(self, tokens: list):
        self.tokens = tokens
        self.pos = 0

    # ------ 基础方法 ------

    def peek(self) -> Tuple[str, Any, int, int]:
        """查看当前 Token，不消耗"""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return ('EOF', None, -1, -1)

    def consume(self, expected_type: str = None,
                expected_value: str = None) -> Tuple[str, Any, int, int]:
        """
        消耗当前 Token。
        如果指定了 expected_type / expected_value，则进行匹配检查，
        不匹配时抛出 ParseError。
        """
        token = self.peek()
        # 类型检查
        if expected_type is not None and token[0] != expected_type:
            raise ParseError(
                "Unexpected token type",
                line=token[2], col=token[3],
                expected=expected_type, got=token[0]
            )
        # 值检查（用于区分不同 KEYWORD）
        if expected_value is not None and token[1] != expected_value:
            raise ParseError(
                "Unexpected token value",
                line=token[2], col=token[3],
                expected=f"{expected_type}({expected_value})",
                got=f"{token[0]}({token[1]})"
            )
        self.pos += 1
        return token

    # ------ 解析入口 ------

    def parse(self) -> ASTNode:
        """入口方法：解析完整的 Feedback 结构"""
        node = self.parse_feedback()
        # 确认所有输入已被消费
        if self.peek()[0] != 'EOF':
            t = self.peek()
            raise ParseError(
                "Extra tokens after parse complete",
                line=t[2], col=t[3],
                expected='EOF', got=f"{t[0]}({t[1]})"
            )
        return node

    # ------ 产生式解析函数 ------

    def parse_feedback(self) -> ASTNode:
        """Feedback → FEEDBACK LBRACE FieldList RBRACE"""
        self.consume('KEYWORD', 'FEEDBACK')
        self.consume('LBRACE')
        field_list_node = self.parse_field_list()
        self.consume('RBRACE')
        # 将 FieldList 的子节点提升为 Feedback 的子节点
        return ASTNode('Feedback', children=field_list_node.children)

    def parse_field_list(self) -> ASTNode:
        """FieldList → Field FieldList | ε
        依据 FIRST(Field) = {SCORE, LEVEL, COMMENT, ERRORS} 判断是否需要展开 Field
        """
        node = ASTNode('FieldList')
        token = self.peek()
        # 判断当前 Token 是否属于 FIRST(Field)
        if token[0] == 'KEYWORD' and token[1] in ('SCORE', 'LEVEL', 'COMMENT', 'ERRORS'):
            field_node = self.parse_field()
            node.children.append(field_node)
            rest_node = self.parse_field_list()
            node.children.extend(rest_node.children)
        # 否则走 ε 产生式，返回空 FieldList
        return node

    def parse_field(self) -> ASTNode:
        """Field → ScoreField | LevelField | CommentBlock | ErrorList
        通过当前 KEYWORD 的值选择对应分支
        """
        token = self.peek()
        if token[0] != 'KEYWORD' or token[1] not in ('SCORE', 'LEVEL', 'COMMENT', 'ERRORS'):
            raise ParseError(
                "Expected field keyword",
                line=token[2], col=token[3],
                expected='SCORE/LEVEL/COMMENT/ERRORS',
                got=f"{token[0]}({token[1]})"
            )
        if token[1] == 'SCORE':
            return self.parse_score_field()
        elif token[1] == 'LEVEL':
            return self.parse_level_field()
        elif token[1] == 'COMMENT':
            return self.parse_comment_block()
        else:  # ERRORS
            return self.parse_error_list()

    def parse_score_field(self) -> ASTNode:
        """ScoreField → SCORE COLON NUMBER SEMICOLON
        扩展挑战：分号可选（容错支持）
        """
        self.consume('KEYWORD', 'SCORE')
        self.consume('COLON')
        num_token = self.consume('NUMBER')
        # 扩展挑战: 分号可选——如果当前是 SEMICOLON 就消费它
        if self.peek()[0] == 'SEMICOLON':
            self.consume('SEMICOLON')
        return ASTNode('Field', value=f'SCORE={num_token[1]}')

    def parse_level_field(self) -> ASTNode:
        """LevelField → LEVEL COLON IDENT SEMICOLON"""
        self.consume('KEYWORD', 'LEVEL')
        self.consume('COLON')
        ident_token = self.consume('IDENT')
        self.consume('SEMICOLON')
        return ASTNode('Field', value=f'LEVEL={ident_token[1]}')

    def parse_comment_block(self) -> ASTNode:
        """CommentBlock → COMMENT LBRACE CommentFieldList RBRACE"""
        self.consume('KEYWORD', 'COMMENT')
        self.consume('LBRACE')
        comment_fields = self.parse_comment_field_list()
        self.consume('RBRACE')
        return ASTNode('Comment', children=comment_fields.children)

    def parse_comment_field_list(self) -> ASTNode:
        """CommentFieldList → CommentField CommentFieldList | ε
        FIRST(CommentField) = {TEXT, SUGGESTION}
        """
        node = ASTNode('CommentFields')
        token = self.peek()
        if token[0] == 'KEYWORD' and token[1] in ('TEXT', 'SUGGESTION'):
            field_node = self.parse_comment_field()
            node.children.append(field_node)
            rest_node = self.parse_comment_field_list()
            node.children.extend(rest_node.children)
        return node

    def parse_comment_field(self) -> ASTNode:
        """CommentField → TextField | SuggestionField"""
        token = self.peek()
        if token[0] == 'KEYWORD':
            if token[1] == 'TEXT':
                return self.parse_text_field()
            elif token[1] == 'SUGGESTION':
                return self.parse_suggestion_field()
        raise ParseError(
            "Expected TEXT or SUGGESTION in comment block",
            line=token[2], col=token[3],
            expected='TEXT/SUGGESTION',
            got=f"{token[0]}({token[1]})"
        )

    def parse_text_field(self) -> ASTNode:
        """TextField → TEXT COLON STRING SEMICOLON"""
        self.consume('KEYWORD', 'TEXT')
        self.consume('COLON')
        str_token = self.consume('STRING')
        self.consume('SEMICOLON')
        return ASTNode('Field', value=f'TEXT={str_token[1]}')

    def parse_suggestion_field(self) -> ASTNode:
        """SuggestionField → SUGGESTION COLON STRING SEMICOLON"""
        self.consume('KEYWORD', 'SUGGESTION')
        self.consume('COLON')
        str_token = self.consume('STRING')
        self.consume('SEMICOLON')
        return ASTNode('Field', value=f'SUGGESTION={str_token[1]}')

    def parse_error_list(self) -> ASTNode:
        """ErrorList → ERRORS LBRACKET ErrorItems RBRACKET"""
        self.consume('KEYWORD', 'ERRORS')
        self.consume('LBRACKET')
        error_items = self.parse_error_items()
        self.consume('RBRACKET')
        return ASTNode('ErrorList', children=error_items.children)

    def parse_error_items(self) -> ASTNode:
        """ErrorItems → ErrorItem ErrorItems | ε
        FIRST(ErrorItem) = {ERROR}
        """
        node = ASTNode('ErrorItems')
        token = self.peek()
        if token[0] == 'KEYWORD' and token[1] == 'ERROR':
            item_node = self.parse_error_item()
            node.children.append(item_node)
            rest_node = self.parse_error_items()
            node.children.extend(rest_node.children)
        return node

    def parse_error_item(self) -> ASTNode:
        """ErrorItem → ERROR LPAREN ErrorParams RPAREN SEMICOLON
        扩展挑战：参数字段顺序任意（容错支持）
        """
        self.consume('KEYWORD', 'ERROR')
        self.consume('LPAREN')
        # 扩展挑战: 解析任意顺序的 key:value 参数
        param_dict = self._parse_error_params_flexible()
        self.consume('RPAREN')
        self.consume('SEMICOLON')
        return ASTNode('Error', value=param_dict)

    def parse_error_params(self) -> ASTNode:
        """ErrorParams → ErrorParam RestParams（标准文法版本）"""
        node = ASTNode('ErrorParams')
        first_param = self.parse_error_param()
        node.children.append(first_param)
        rest_node = self.parse_rest_params()
        node.children.extend(rest_node.children)
        return node

    def _parse_error_params_flexible(self) -> Dict[str, Any]:
        """
        扩展挑战：支持任意顺序的 line/type/msg 参数。
        不再按固定顺序解析，而是收集所有 key:value 对存入字典。
        这与标准文法的 ErrorParams 和 RestParams 对应，
        但允许参数以任意 KEY:VALUE 顺序出现。
        """
        params = {}
        # 至少有一个参数
        key, val = self._parse_single_param()
        params[key] = val
        # 后续参数以逗号分隔
        while self.peek()[0] == 'COMMA':
            self.consume('COMMA')
            key, val = self._parse_single_param()
            if key in params:
                raise ParseError(
                    f"Duplicate parameter key '{key}'",
                    line=self.peek()[2],
                    expected='unique parameter key',
                    got=key
                )
            params[key] = val
        return params

    def _parse_single_param(self) -> Tuple[str, Any]:
        """解析单个参数: IDENT COLON ErrorValue"""
        key_token = self.consume('IDENT')
        self.consume('COLON')
        val_token = self._parse_error_value()
        return key_token[1], val_token

    def parse_rest_params(self) -> ASTNode:
        """RestParams → COMMA ErrorParam RestParams | ε"""
        node = ASTNode('RestParams')
        token = self.peek()
        if token[0] == 'COMMA':
            self.consume('COMMA')
            param_node = self.parse_error_param()
            node.children.append(param_node)
            rest_node = self.parse_rest_params()
            node.children.extend(rest_node.children)
        return node

    def parse_error_param(self) -> ASTNode:
        """ErrorParam → IDENT COLON ErrorValue"""
        ident_token = self.consume('IDENT')
        self.consume('COLON')
        val = self._parse_error_value()
        return ASTNode('Param', value=(ident_token[1], val))

    def _parse_error_value(self) -> Any:
        """ErrorValue → NUMBER | STRING | IDENT
        返回参数的实际值（而非 Token 四元组）
        """
        token = self.peek()
        if token[0] == 'NUMBER':
            self.consume('NUMBER')
            return token[1]
        elif token[0] == 'STRING':
            self.consume('STRING')
            return token[1]
        elif token[0] == 'IDENT':
            self.consume('IDENT')
            return token[1]
        else:
            raise ParseError(
                "Expected NUMBER, STRING, or IDENT as parameter value",
                line=token[2], col=token[3],
                expected='NUMBER/STRING/IDENT',
                got=token[0]
            )


# ============================================================
# 便捷函数
# ============================================================
def parse_text(text: str) -> ASTNode:
    """一站式解析函数：文本 → AST"""
    lexer = Lexer(text)
    tokens = lexer.tokenize()
    parser = EduParser(tokens)
    return parser.parse()


def print_ast_from_text(text: str) -> None:
    """解析文本并打印 AST 树形结构"""
    ast_root = parse_text(text)
    ast_root.pprint()


# ============================================================
# 主入口：演示
# ============================================================
if __name__ == '__main__':
    sample = '''FEEDBACK {
    SCORE: 85;
    LEVEL: medium;
    COMMENT {
        TEXT: "逻辑清晰，但边界处理不足";
        SUGGESTION: "增加空指针检查";
    }
    ERRORS [
        ERROR(line:12, type:runtime, msg:"NullPointerException");
        ERROR(line:27, type:logic, msg:"边界条件错误");
    ]
}'''

    print("=" * 50)
    print("EduAssist Parser Demo")
    print("=" * 50)

    # 词法分析
    lexer = Lexer(sample)
    tokens = lexer.tokenize()
    print("\n[词法分析] Tokens:")
    for t in tokens[:-1]:  # 排除 EOF
        print(f"  {t[0]:12s} = {t[1]!r}")

    # 语法分析
    print("\n[语法分析] AST:")
    parser = EduParser(tokens)
    ast_root = parser.parse()
    ast_root.pprint()

    # 容错测试：SCORE 省略分号
    print("\n" + "=" * 50)
    print("容错测试: SCORE省略分号")
    print("=" * 50)
    sample_no_semi = 'FEEDBACK { SCORE: 90 LEVEL: hard; }'
    ast2 = parse_text(sample_no_semi)
    ast2.pprint()

    # 容错测试：ERROR参数乱序
    print("\n" + "=" * 50)
    print("容错测试: ERROR参数乱序")
    print("=" * 50)
    sample_reorder = 'FEEDBACK { ERRORS [ ERROR(type:runtime, msg:"Oops", line:5); ] }'
    ast3 = parse_text(sample_reorder)
    ast3.pprint()
