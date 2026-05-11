# EduAssist 语法分析器 — 编译原理课程作业

面向教育场景的大模型输出语法分析器设计与实现。

## 项目结构

.
├── edu_parser.py # 任务二：递归下降语法分析器 + 词法分析器
├── evaluate.py # 任务三：端到端系统集成（LLM API + Parser）
├── test_cases/ # 测试用例
│ ├── correct_1.txt # 正确输入 1
│ ├── correct_2.txt # 正确输入 2
│ ├── error_1.txt # 含语法错误的输入 1
│ ├── error_2.txt # 含语法错误的输入 2
│ └── llm_real_output.txt # LLM 真实输出示例
└── README.md # 本文件


## 环境依赖

```bash
pip install openai
```


> 仅 `evaluate.py` 需要 `openai` 库。`edu_parser.py` 可独立运行，无额外依赖。

## API 密钥配置

支持通过环境变量配置 LLM API 参数：


# 智谱 GLM（推荐）

export LLM_API_BASE_URL="https://open.bigmodel.cn/api/paas/v4"
export LLM_API_KEY="your-zhipu-api-key"
export LLM_API_MODEL="glm-4-flash"

# 通义千问

export LLM_API_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export LLM_API_KEY="your-dashscope-api-key"
export LLM_API_MODEL="qwen-turbo"

# Moonshot

export LLM_API_BASE_URL="https://api.moonshot.cn/v1"
export LLM_API_KEY="your-moonshot-api-key"
export LLM_API_MODEL="moonshot-v1-8k"
