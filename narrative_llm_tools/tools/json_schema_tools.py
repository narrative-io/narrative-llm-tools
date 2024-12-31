import functools
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from narrative_llm_tools.rest_api_client.rest_api_client import RestApiClient
from narrative_llm_tools.rest_api_client.types import RestApiConfig

import logging

logger = logging.getLogger(__name__)
    
class NameProperty(BaseModel):
    """
    Property for the name of the tool.  We keep it in this semi-strange format to 
    maintain compatibility with JSON Schema.
    """
    type: Literal["string"]
    enum: List[str]
    description: Optional[str] = None
    
    model_config = {
        "frozen": True
    }

class ParametersProperty(BaseModel):
    type: Literal["object"]
    properties: Dict[str, Any]
    required: Optional[List[str]] = None
    additionalProperties: Optional[bool] = None
    
    model_config = {
        "frozen": True
    }

class ToolProperties(BaseModel):
    name: NameProperty
    parameters: ParametersProperty
    
    model_config = {
        "frozen": True
    }

class ToolSchema(BaseModel):
    """
    A tool schema is a JSON Schema object that describes a tool.  Every tool must have
    a name and parameters.  The parameters are a JSON Schema object that describes the
    parameters for the tool.
    """
    type: Literal["object"]
    required: List[Literal["name", "parameters"]]
    restApi: Optional[RestApiConfig] = Field(default=None, exclude=True)
    properties: ToolProperties
    additionalProperties: Optional[bool] = None
    
    model_config = {
        "json_schema_extra": {"exclude": ["restApi"]},
        "frozen": True
    }
    
    def __hash__(self) -> int:
        return hash((
            self.type,
            tuple(self.required),
            self.restApi,
            tuple(self.properties.name.enum),
            tuple(self.properties.parameters.required) if self.properties.parameters.required else None,
            self.properties.parameters.additionalProperties,
            self.additionalProperties
        ))
        
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ToolSchema):
            return False
        return (
            self.type == other.type and
            self.required == other.required and
            self.restApi == other.restApi and
            self.properties.name.enum == other.properties.name.enum and
            self.properties.parameters.required == other.properties.parameters.required and
            self.properties.parameters.additionalProperties == other.properties.parameters.additionalProperties and
            self.additionalProperties == other.additionalProperties
        )
    
    @classmethod
    def user_response_tool(cls) -> "ToolSchema":
        return cls(
            type="object",
            required=["name", "parameters"],
            properties=ToolProperties(
                name=NameProperty(
                    type="string",
                    enum=["respond_to_user"],
                    description="When you are ready to respond to the user, use this tool"
                ),
                parameters=ParametersProperty(
                    type="object",
                    properties={
                        "response": {
                            "type": "string",
                            "description": "The response to the user"
                        }
                    },
                    required=["response"],
                    additionalProperties=False
                )
            )
        )

class Tool(BaseModel):
    name: str
    parameters: Dict[str, Any]

class ItemsSchema(BaseModel):
    anyOf: List[ToolSchema]
    
    

class JsonSchemaTools(BaseModel):
    type: Literal["array"]
    items: ItemsSchema
    
    model_config = {
        "frozen": True
    }
    
    def __hash__(self) -> int:
        return hash((
            self.type,
            tuple(self.items.anyOf)  
        ))
        
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, JsonSchemaTools):
            return False
        return (
            self.type == other.type and
            self.items.anyOf == other.items.anyOf
        )

    def get_rest_apis(self) -> Dict[str, "RestApiClient"]:
        apis: Dict[str, RestApiClient] = {}
        for tool in self.items.anyOf:
            if tool.restApi:
                for name in tool.properties.name.enum:
                    apis[name] = RestApiClient(name=name, config=tool.restApi)
        return apis
    
    def remove_tool_by_name(self, name: str) -> "JsonSchemaTools":
        # Create new list of tools excluding the matching one
        new_tools = [
            tool for tool in self.items.anyOf
            if not (len(tool.properties.name.enum) == 1 and tool.properties.name.enum[0] == name)
        ]
        
        # Create new tools with filtered enum values
        new_tools = [
            tool.model_copy(update={
                "properties": tool.properties.model_copy(update={
                    "name": tool.properties.name.model_copy(update={
                        "enum": [n for n in tool.properties.name.enum if n != name]
                    })
                })
            })
            for tool in new_tools
        ]
        
        # Return new instance with updated tools
        return self.model_copy(update={
            "items": self.items.model_copy(update={
                "anyOf": new_tools
            })
        })
        
    def remove_rest_api_tools(self) -> "JsonSchemaTools":
        logger.debug(f"Original tools: {self.items.anyOf}")
        new_tools = [
            tool for tool in self.items.anyOf
            if not tool.restApi
        ]
        logger.debug(f"Filtered tools: {new_tools}")
        return self.model_copy(update={
            "items": self.items.model_copy(update={
                "anyOf": new_tools
            })
        })
        
    def clear_tools(self) -> None:
        self.items.anyOf = []
        
    def with_user_response_tool(self) -> "JsonSchemaTools":
        # Create a new list combining existing tools and the user response tool
        new_tools = self.items.anyOf + [ToolSchema.user_response_tool()]
        return self.model_copy(update={
            "items": self.items.model_copy(update={
                "anyOf": new_tools
            })
        })
        
    @classmethod
    def only_user_response_tool(cls) -> "JsonSchemaTools":
        return cls(
            type="array",
            items=ItemsSchema(anyOf=[ToolSchema.user_response_tool()])
        )
            