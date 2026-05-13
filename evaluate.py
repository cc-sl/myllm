# evaluate.py
"""
EduAssist 端到端系统集成
数据流: 学生代码 → Prompt模板填充 → LLM API调用 → 原始文本 → 词法分析 → Parser → AST → 结构化JSON

依赖:
  pip install openai  (兼容 OpenAI 接口的国内 LLM 服务)
"""

import json
import os
import sys
from typing import Dict, Any, Optional, Tuple

# 导入自定义的词法分析器和语法分析器
from edu_parser import Lexer, EduParser, ASTNode, ParseError


# main.py
from config import API_BASE_URL, API_KEY, API_MODEL

# print(API_BASE_URL)
# print(API_KEY)
# print(API_MODEL)


# ============================================================
# 3.1 Prompt 设计
# ============================================================
SYSTEM_PROMPT = """你是一名编程课AI助教。你的任务是对学生提交的Java代码进行批改，并严格按照以下格式输出结构化反馈：

FEEDBACK {
  SCORE: 分数;
  LEVEL: 难度等级;
  COMMENT {
    TEXT: "评语内容";
    SUGGESTION: "改进建议";
  }
  ERRORS [
    ERROR(line:行号, type:错误类型, msg:"错误信息");
  ]
}

格式规则（必须严格遵守）：
1. FEEDBACK 必须是顶层结构，用花括号包围
2. SCORE 后接冒号和0-100的整数，以分号结尾
3. LEVEL 后接冒号和关键字(hard/medium/easy)，以分号结尾
4. COMMENT 块用花括号包围，包含 TEXT 和 SUGGESTION 两个子字段，值为双引号字符串，以分号结尾
5. ERRORS 用方括号包围，每个 ERROR 格式为 ERROR(line:数字, type:标识符, msg:"字符串");
6. ERROR 中的三个参数 line/type/msg 以逗号分隔
7. 如果代码无错误，ERRORS 列表可以为空: ERRORS []
8. 不要输出任何额外文本，只输出上述格式

以下是正确输出的示例：

FEEDBACK {
SCORE: 75;
LEVEL: medium;
COMMENT {
TEXT: "算法思路正确，但存在数组越界风险";
SUGGESTION: "在访问数组前添加长度检查";
}
ERRORS [
ERROR(line:8, type:runtime, msg:"ArrayIndexOutOfBoundsException");
ERROR(line:15, type:logic, msg:"循环条件错误");
]
}
"""

# User Prompt 模板
USER_PROMPT_TEMPLATE = """请批改以下学生提交的Java代码，严格按照指定的结构化格式输出反馈：

```java
{code}
```"""

MAX_RETRIES = 1  # ParseError 时的最大重试次数


# ============================================================
# LLM API 调用
# ============================================================
def call_llm(code_snippet: str) -> str:
    """
    调用 LLM API，获取结构化反馈文本。
    使用 OpenAI 兼容接口。
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("错误: 请安装 openai 库: pip install openai", file=sys.stderr)
        sys.exit(1)

    if not API_KEY:
        raise ValueError(
            "请设置 LLM_API_KEY 环境变量，例如:\n"
            "  export LLM_API_KEY='your-api-key'"
        )

    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
    )

    user_message = USER_PROMPT_TEMPLATE.format(code=code_snippet)

    response = client.chat.completions.create(
        model=API_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,  # 较低温度以提高格式稳定性
        max_tokens=1024,
    )

    return response.choices[0].message.content


# ============================================================
# AST → 结构化字典转换
# ============================================================
def ast_to_dict(ast: ASTNode) -> Dict[str, Any]:
    """
    将 AST 转换为包含 score, level, comment, errors 字段的结构化字典。
    """
    result = {
        "score": None,
        "level": None,
        "comment": {"text": "", "suggestion": ""},
        "errors": []
    }

    for child in ast.children:
        if child.node_type == 'Field':
            val = child.value
            if val.startswith('SCORE='):
                result["score"] = int(val.split('=', 1)[1])
            elif val.startswith('LEVEL='):
                result["level"] = val.split('=', 1)[1]

        elif child.node_type == 'Comment':
            for sub in child.children:
                if sub.node_type == 'Field':
                    val = sub.value
                    if val.startswith('TEXT='):
                        result["comment"]["text"] = val.split('=', 1)[1]
                    elif val.startswith('SUGGESTION='):
                        result["comment"]["suggestion"] = val.split('=', 1)[1]

        elif child.node_type == 'ErrorList':
            for err in child.children:
                if err.node_type == 'Error' and isinstance(err.value, dict):
                    result["errors"].append({
                        "line": err.value.get("line"),
                        "type": err.value.get("type"),
                        "msg": err.value.get("msg"),
                    })

    return result


# ============================================================
# 解析 LLM 输出文本
# ============================================================
def parse_llm_output(text: str) -> ASTNode:
    """
    对 LLM 原始输出文本进行词法分析和语法分析，返回 AST。
    自动提取文本中的 FEEDBACK {...} 块（处理 LLM 可能添加的 markdown 包裹）。
    """
    # 尝试提取 FEEDBACK 块（处理 markdown 代码块包裹）
    feedback_text = text.strip()

    # 去除 markdown 代码块标记
    if feedback_text.startswith('```'):
        lines = feedback_text.split('\n')
        # 去掉首尾的 ``` 行
        start_idx = 0
        end_idx = len(lines) - 1
        for i, line in enumerate(lines):
            if line.strip().startswith('```') and i == start_idx:
                start_idx = i + 1
            elif line.strip().startswith('```'):
                end_idx = i
                break
        feedback_text = '\n'.join(lines[start_idx:end_idx])

    # 词法分析
    lexer = Lexer(feedback_text)
    tokens = lexer.tokenize()

    # 语法分析
    parser = EduParser(tokens)
    ast = parser.parse()
    return ast


# ============================================================
# 3.2 端到端评估函数
# ============================================================
def evaluate_submission(code_snippet: str) -> Dict[str, Any]:
    """
    端到端评估函数。
    输入: 学生代码片段
    输出: 包含 score, level, comment, errors 字段的字典

    当 Parser 抛出 ParseError 时，至多重试1次（附加格式错误提示）。
    """
    retry_count = 0
    last_error = None

    while retry_count <= MAX_RETRIES:
        try:
            # 1. 调用 LLM
            if retry_count == 0:
                raw_output = call_llm(code_snippet)
                print("\nLLM 原始输出")
                print(raw_output)
            else:
                # 重试时附加格式错误提示
                error_hint = (
                    f"你的上一次输出格式有误，错误信息: {last_error}。\n"
                    "请严格按照以下格式输出，不要添加任何额外内容:\n"
                    "FEEDBACK { SCORE: 数字; LEVEL: 等级; COMMENT { TEXT: \"...\"; SUGGESTION: \"...\"; } ERRORS [ ... ] }"
                )
                # 使用 OpenAI 接口进行重试
                from openai import OpenAI
                client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
                user_msg = USER_PROMPT_TEMPLATE.format(code=code_snippet) + "\n\n" + error_hint
                response = client.chat.completions.create(
                    model=API_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.1,
                    max_tokens=1024,
                )
                raw_output = response.choices[0].message.content

            # 2. 解析 LLM 输出
            ast = parse_llm_output(raw_output)

            # 3. 转换为结构化字典
            result = ast_to_dict(ast)
            result["_raw_output"] = raw_output
            result["_parse_success"] = True
            result["_retried"] = (retry_count > 0)
            return result

        except ParseError as e:
            last_error = str(e)
            retry_count += 1
            print(f"[ParseError] 第{retry_count}次尝试失败: {e}", file=sys.stderr)

    # 重试仍失败
    return {
        "score": None,
        "level": None,
        "comment": {"text": "", "suggestion": ""},
        "errors": [],
        "_raw_output": raw_output if 'raw_output' in dir() else "",
        "_parse_success": False,
        "_retried": True,
        "_error": str(last_error),
    }


# ============================================================
# 本地测试（不调用 API，仅测试 Parser）
# ============================================================
def evaluate_local(feedback_text: str) -> Dict[str, Any]:
    """
    本地测试函数：直接解析给定的反馈文本，无需调用 LLM API。
    用于测试 Parser 的正确性。
    """
    ast = parse_llm_output(feedback_text)
    ast.pprint()
    return ast_to_dict(ast)


# ============================================================
# 稳定性测试辅助函数
# ============================================================
def test_stability(code_snippets: list) -> dict:
    """
    稳定性测试：对多段学生代码调用 LLM，统计格式合规率。
    """
    results = []
    success_count = 0
    total = len(code_snippets)

    for i, code in enumerate(code_snippets):
        print(f"\n--- 测试用例 {i + 1} ---")
        try:
            result = evaluate_submission(code)
            if result.get("_parse_success"):
                success_count += 1
                print(f"  解析成功 ✓ 分数: {result['score']}, 等级: {result['level']}")
            else:
                print(f"  解析失败 ✗ 错误: {result.get('_error', 'unknown')}")
            results.append(result)
        except Exception as e:
            print(f"  系统异常 ✗ {e}")
            results.append({"_parse_success": False, "_error": str(e)})

    compliance_rate = success_count / total if total > 0 else 0
    summary = {
        "total": total,
        "success": success_count,
        "compliance_rate": compliance_rate,
        "details": results,
    }
    print(f"\n稳定性测试结果: {success_count}/{total} 合规, 合规率 = {compliance_rate:.1%}")
    return summary


# ============================================================
# 主入口
# ============================================================
if __name__ == '__main__':
    import argparse
    import json  # 确保导入json模块，原代码用到了

    parser_arg = argparse.ArgumentParser(description='EduAssist 端到端批改系统')
    parser_arg.add_argument('--local', action='store_true',
                            help='仅本地测试 Parser（不调用 LLM API）')
    parser_arg.add_argument('--api-test', action='store_true',
                            help='调用 LLM API 进行稳定性测试')
    parser_arg.add_argument('--input', type=str, default=None,
                            help='输入学生代码文件路径')
    args = parser_arg.parse_args()

    # 核心修改：默认启动本地解析 
    # 如果没有传入任何参数，自动启用本地模式
    if not any([args.local, args.api_test, args.input]):
        args.local = True

    if args.local:
        # 本地测试
        test_input_1 = '''FEEDBACK {
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
        print("=== 本地解析测试 ===")
        result = evaluate_local(test_input_1)
        print("\n结构化结果:")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.api_test:
        # API 稳定性测试
        test_codes = [
            # 代码1: 有空指针风险+数组越界
            """public int findMax(int[] arr) {
    int max = arr[0];
    for (int i = 1; i <= arr.length; i++) {
        if (arr[i] > max) max = arr[i];
    }
    return max;
}""",
            # 代码2: 有逻辑错误(素数判断效率极低)
            """public boolean isPrime(int n) {
    if (n < 2) return false;
    for (int i = 2; i < n; i++) {
        if (n % i == 0) return false;
    }
    return true;
}""",
            # 代码3: 基本正确(阶乘计算)
            """public int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}""",
            # 代码4: 除零异常风险
            """public double divide(int a, int b) {
    return a / b;
}""",
            # 代码5: 空指针异常(字符串操作)
            """public int getStringLength(String str) {
    return str.length();
}""",
            # 代码6: 死循环风险
            """public void infiniteLoop() {
    int i = 0;
    while (i >= 0) {
        i++;
    }
}""",
            # 代码7: 类型转换错误(向下转型)
            """public int castToInt(Object obj) {
    return (Integer) obj;
}""",
            # 代码8: 基本正确(两数求和)
            """public int add(int a, int b) {
    return a + b;
}""",
            # 代码9: 数组越界(字符串索引)
            """public char getFirstChar(String str) {
    return str.charAt(0);
}""",
            # 代码10: 有逻辑错误(斐波那契数列)
            """public int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}"""
        ]
        test_stability(test_codes)

    elif args.input:
        # 从文件读取学生代码
        with open(args.input, 'r', encoding='utf-8') as f:
            code = f.read()
        result = evaluate_submission(code)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        print("使用方式:")
        print("  python evaluate.py --local       # 本地 Parser 测试")
        print("  python evaluate.py --api-test     # LLM API 稳定性测试")
        print("  python evaluate.py --input code.txt  # 端到端评估")