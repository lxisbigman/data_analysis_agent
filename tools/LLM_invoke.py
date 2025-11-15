###LLM_invoke.py
import requests
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import List, Optional, Any, Dict, Union
import logging
from env.LLM_env import DEEPSEEK_URL, DEEPSEEK_API_KEY, qwen3_url

# 设置日志
logger = logging.getLogger(__name__)


class DeepSeekLLM(BaseChatModel):
    """DeepSeek大模型接口，支持备用API降级"""

    api_url: str = DEEPSEEK_URL
    api_key: str = DEEPSEEK_API_KEY
    temperature: float = 0.7
    max_tokens: int = 10000
    timeout: int = 60
    back_up_url: str = qwen3_url

    @property
    def _llm_type(self) -> str:
        return "deepseek"

    def _format_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """格式化消息为API所需格式"""
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                logger.warning(f"未知的消息类型: {type(msg)}，默认使用user角色")
                role = "user"

            formatted_messages.append({
                "role": role,
                "content": msg.content
            })
        return formatted_messages

    def _call_api(self, url: str, payload: Dict[str, Any]) -> str:
        """调用API并返回内容"""
        try:
            logger.info(f"调用API: {url}")
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            if "choices" not in result or not result["choices"]:
                raise ValueError("API响应中缺少choices字段")

            message = result["choices"][0].get("message", {})
            content = message.get("content", "")

            if not content:
                logger.warning("API返回空内容")

            return content

        except requests.exceptions.Timeout:
            logger.error(f"API请求超时: {url}")
            raise ValueError(f"API请求超时，请检查网络连接或调整timeout设置")
        except requests.exceptions.ConnectionError:
            logger.error(f"连接错误: {url}")
            raise ValueError(f"无法连接到API服务，请检查网络连接")
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP错误 {response.status_code}: {response.text}")
            raise ValueError(f"API请求失败: {e}")
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"响应解析错误: {e}")
            raise ValueError(f"API响应格式错误: {e}")
        except Exception as e:
            logger.error(f"未知错误: {e}")
            raise ValueError(f"API调用失败: {e}")

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        """生成聊天回复 - 包含完整的备用API逻辑"""
        formatted_messages = self._format_messages(messages)

        # 构建请求载荷
        payload = {
            "messages": formatted_messages,
            "temperature": kwargs.get('temperature', self.temperature),
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "stream": False
        }

        # 添加停止词
        if stop:
            payload["stop"] = stop

        # 合并其他参数
        payload.update(kwargs)

        # 首先尝试主API（DeepSeek）
        try:
            logger.info("尝试主API (DeepSeek)...")
            content = self._call_api(self.api_url, payload)
            logger.info("主API调用成功")

        except Exception as primary_error:
            logger.warning(f"主API调用失败: {primary_error}")

            # 主API失败时，尝试备用API（qwen3）
            if self.back_up_url:
                logger.info("尝试备用API (qwen3)...")
                try:
                    content = self._call_api(self.back_up_url, payload)
                    logger.info("备用API调用成功")
                except Exception as backup_error:
                    logger.error(f"备用API也失败: {backup_error}")
                    # 两个API都失败，抛出详细错误
                    raise ValueError(
                        f"所有API调用均失败:\n"
                        f"主API错误: {primary_error}\n"
                        f"备用API错误: {backup_error}"
                    )
            else:
                # 没有备用API，直接抛出主API错误
                raise ValueError(f"主API调用失败且未设置备用API: {primary_error}")

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )

    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any):
        """流式生成（如需实现）"""
        raise NotImplementedError("DeepSeekLLM暂不支持流式生成")

    def invoke(self, input: Union[str, Dict, List[BaseMessage]], **kwargs: Any) -> Any:
        """调用接口的统一方法"""
        # 解析输入
        if isinstance(input, str):
            messages = [HumanMessage(content=input)]
        elif isinstance(input, dict) and "messages" in input:
            messages = []
            for msg in input["messages"]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
                elif role == "system":
                    messages.append(SystemMessage(content=content))
                else:
                    logger.warning(f"未知的角色: {role}，默认使用HumanMessage")
                    messages.append(HumanMessage(content=content))
        elif isinstance(input, list) and all(isinstance(m, BaseMessage) for m in input):
            messages = input
        else:
            raise ValueError(f"不支持的输入类型: {type(input)}，支持str、dict或List[BaseMessage]")

        # 调用生成方法
        return self._generate(messages, **kwargs)

    def get_num_tokens(self, text: str) -> int:
        """估算token数量（简化实现）"""
        # 这里可以使用更精确的tokenizer，目前使用简单估算
        return len(text) // 4

    def with_parameters(self, **kwargs) -> 'DeepSeekLLM':
        """返回带有新参数的实例副本"""
        import copy
        new_instance = copy.copy(self)
        for key, value in kwargs.items():
            if hasattr(new_instance, key):
                setattr(new_instance, key, value)
        return new_instance


# 使用示例和测试
if __name__ == "__main__":
    # 初始化模型
    llm = DeepSeekLLM()

    # 测试不同输入方式
    try:
        # 方式1: 字符串输入
        print("测试字符串输入...")
        result1 = llm.invoke("你好，请介绍一下你自己")
        print("字符串输入结果:", result1.generations[0].message.content[:100] + "...")

        # 方式2: 消息列表输入
        print("测试消息列表输入...")
        messages = [
            SystemMessage(content="你是一个专业的AI助手"),
            HumanMessage(content="什么是机器学习？")
        ]
        result2 = llm.invoke(messages)
        print("消息列表结果:", result2.generations[0].message.content[:100] + "...")

        print("所有测试通过！")

    except Exception as e:
        print(f"测试失败: {e}")