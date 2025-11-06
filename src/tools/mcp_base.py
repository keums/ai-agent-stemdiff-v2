from typing import Any, Dict, List, Optional, Type, Callable
from pydantic import BaseModel
import logging
import asyncio
import pathlib
from functools import wraps
from dotenv import load_dotenv

# .env 파일 로드 (프로젝트 전체에서 한 번만)
env_path = pathlib.Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

logger = logging.getLogger(__name__)

class MCPToolRegistry:
    """MCP 도구 등록 및 관리 클래스"""
    
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}
    
    def register_tool(self, name: str, func: Callable, description: str, 
                     input_schema: Type[BaseModel], output_schema: Type[BaseModel]):
        """도구를 레지스트리에 등록"""
        self._tools[name] = {
            'function': func,
            'description': description,
            'input_schema': input_schema,
            'output_schema': output_schema
        }
        logger.info(f"Tool '{name}' registered successfully")
    
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """등록된 도구 정보 반환"""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """등록된 모든 도구 이름 반환"""
        return list(self._tools.keys())
    
    def get_tool_schemas(self) -> Dict[str, Dict[str, Any]]:
        """모든 도구의 스키마 정보 반환"""
        schemas = {}
        for name, tool_info in self._tools.items():
            schemas[name] = {
                'description': tool_info['description'],
                'input_schema': tool_info['input_schema'].model_json_schema(),
                'output_schema': tool_info['output_schema'].model_json_schema()
            }
        return schemas

# 전역 도구 레지스트리
tool_registry = MCPToolRegistry()

def tool(name: str, description: str, input_schema: Type[BaseModel], 
         output_schema: Type[BaseModel]):
    """
    MCP 스타일 도구 데코레이터
    
    Args:
        name: 도구 이름
        description: 도구 설명
        input_schema: 입력 스키마 (Pydantic 모델)
        output_schema: 출력 스키마 (Pydantic 모델)
    """
    def decorator(func: Callable) -> Callable:
        # 도구를 레지스트리에 등록
        tool_registry.register_tool(name, func, description, input_schema, output_schema)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            """비동기 래퍼"""
            try:
                logger.info(f"Executing tool: {name}")
                
                # 입력 파라미터 검증
                # if args and isinstance(args[0], dict):
                #     # 딕셔너리 형태의 입력인 경우
                #     input_data = input_schema(**args[0])
                # elif kwargs:
                #     # 키워드 인자인 경우
                #     input_data = input_schema(**kwargs)
                # else:
                #     raise ValueError("Invalid input parameters")
                
                # 함수 실행 (sync/async 모두 지원)
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # 출력 검증
                if isinstance(result, output_schema):
                    return result
                elif isinstance(result, dict):
                    return output_schema(**result)
                else:
                    logger.warning(f"Tool {name} returned unexpected format: {type(result)}")
                    return result
                    
            except Exception as e:
                logger.error(f"Error executing tool {name}: {str(e)}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            """동기 래퍼"""
            try:
                logger.info(f"Executing tool: {name}")
                
                # 입력 파라미터 검증
                if args and isinstance(args[0], dict):
                    input_data = input_schema(**args[0])
                elif kwargs:
                    input_data = input_schema(**kwargs)
                else:
                    raise ValueError("Invalid input parameters")
                
                # 함수 실행
                result = func(input_data.model_dump())
                
                # 출력 검증
                if isinstance(result, output_schema):
                    return result
                elif isinstance(result, dict):
                    return output_schema(**result)
                else:
                    logger.warning(f"Tool {name} returned unexpected format: {type(result)}")
                    return result
                    
            except Exception as e:
                logger.error(f"Error executing tool {name}: {str(e)}")
                raise
        
        # 원본 함수가 async인지 확인하여 적절한 래퍼 반환
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

class MCPBaseTool:
    """MCP 도구의 기본 클래스"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def validate_input(self, input_data: Dict[str, Any], schema: Type[BaseModel]) -> BaseModel:
        """입력 데이터 검증"""
        try:
            return schema(**input_data)
        except Exception as e:
            logger.error(f"Input validation failed for {self.name}: {str(e)}")
            raise ValueError(f"Invalid input for {self.name}: {str(e)}")
    
    def format_output(self, result: Any, schema: Type[BaseModel]) -> BaseModel:
        """출력 데이터 포맷팅"""
        try:
            if isinstance(result, schema):
                return result
            elif isinstance(result, dict):
                return schema(**result)
            else:
                raise ValueError(f"Unexpected output format: {type(result)}")
        except Exception as e:
            logger.error(f"Output formatting failed for {self.name}: {str(e)}")
            raise ValueError(f"Invalid output for {self.name}: {str(e)}") 