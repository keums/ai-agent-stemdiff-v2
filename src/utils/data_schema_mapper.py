"""
DataSchema 매핑 및 저장 시스템
도구 결과를 dataSchema.json 형태로 변환하고 저장하는 기능을 제공
"""

import json
import logging
import os
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# from models import ContextSong  # Currently unused

logger = logging.getLogger(__name__)


class DataSchemaMapper:
    """
    도구 결과를 dataSchema.json 형태로 매핑하는 클래스
    """

    def __init__(self):
        """매핑 규칙 초기화"""
        self.tool_mappings = {
            "user_prompt": self._map_user_prompt,
            "generate_music_info": self._map_generate_music_info,
            "search_stems_from_es": self._map_search_stems_from_es,
            "reply_orchestrator": self._map_reply_orchestrator,
            "generate_stem_diff": self._map_generate_stem_diff,
        }

    def _to_camel_case(self, snake_str: str) -> str:
        """스네이크케이스를 카멜케이스로 변환"""
        components = snake_str.split("_")
        return components[0] + "".join(word.capitalize() for word in components[1:])

    def _convert_dict_keys_to_camel_case(self, data: Any) -> Any:
        """딕셔너리의 키를 카멜케이스로 변환 (재귀적으로 처리)"""
        if isinstance(data, dict):
            # 특별한 섹션 데이터 형태인지 확인 (단일 키-값 쌍)
            if len(data) == 1:
                key, value = next(iter(data.items()))
                # 섹션 이름과 역할 형태인지 확인 (키가 대문자 알파벳 1글자, 값이 문자열)
                if (
                    isinstance(key, str)
                    and len(key) == 1
                    and key.isalpha()
                    and key.isupper()
                    and isinstance(value, str)
                ):
                    # 섹션 데이터는 변환하지 않고 그대로 반환
                    return data

            return {
                (
                    self._to_camel_case(k) if isinstance(k, str) else k
                ): self._convert_dict_keys_to_camel_case(v)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._convert_dict_keys_to_camel_case(item) for item in data]
        else:
            return data

    def create_base_schema(
        self, task_id: str, session_uuid: Optional[str] = None
    ) -> Dict[str, Any]:
        """기본 dataSchema 구조 생성"""
        if not session_uuid:
            session_uuid = str(uuid.uuid4())

        # request_uuid = str(uuid.uuid4())

        return {
            "request": [
                {
                    "uuid": task_id,
                    # "taskId": task_id,
                    "sessionUuid": session_uuid,
                    "speaker": "user",
                    "dataType": "text",
                    "turnIndex": 1,
                    "thumb": False,
                    "status": "inProgress",
                    "tokenUsageId": [],
                    "requestUuid": None,
                    "processingTimeMs": 0,
                    "schemaVersion": "1.2",
                    "errorMessage": "",
                    "createdAt": datetime.now().isoformat(),
                    "medias": [],
                    "chatMessage": None,
                    "context": {
                        "previousContext": [],
                    },
                    "requestInformation": {
                        "globalMusicInformation": None,
                        "stemPrompts": None,
                        "contextSongInfo": {},
                        # "currentMixIndex": 0,
                        # "promptStemInfo": None,
                    },
                    "answers": [],
                }
            ]
        }

    def map_tool_output(
        self, tool_name: str, output_data: Dict[str, Any], schema_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """도구 출력을 스키마 데이터에 매핑"""
        if tool_name in self.tool_mappings:
            return self.tool_mappings[tool_name](output_data, schema_data)
        return schema_data

    def _map_user_prompt(
        self, output_data: Dict[str, Any], schema_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        request = schema_data["request"][0]
        turn_index = output_data.get("turn_index")
        if not request.get("chatMessage"):
            request["chatMessage"] = {
                "uuid": str(uuid.uuid4()),
                "dialogUuid": request["uuid"],
                "createdAt": datetime.now().isoformat(),
            }
        request["turnIndex"] = 1
        if turn_index != 1:
            request["turnIndex"] = turn_index + 1

        request["chatMessage"]["chatText"] = output_data.get("user_prompt")
        request["chatMessage"]["intentFocusedPrompt"] = output_data.get(
            "intent_focused_prompt"
        )
        return schema_data

    def _map_generate_music_info(
        self, output_data: Dict[str, Any], schema_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        request = schema_data["request"][0]
        answer_uuid = str(uuid.uuid4())
        request["answers"] = [
            {
                "uuid": answer_uuid,
                "sessionUuid": request["sessionUuid"],
                "speaker": "assistant",
                "dataType": "mix",
                "chatMessage": {
                    "uuid": str(uuid.uuid4()),
                    "dialogUuid": answer_uuid,
                    "createdAt": datetime.now().isoformat(),
                    "chatText": "",
                },
                "createdAt": datetime.now().isoformat(),
                "turnIndex": request["turnIndex"] + 1,
                "requestUuid": request["uuid"],
                "thumb": False,
                "status": "inProgress",
                "tokenUsageId": [],
                "processingTimeMs": 0,
                "schemaVersion": "1.2",
                "errorMessage": "",
                # "mix": {
                #     "mixData": {
                #         "stems": [],
                #     },
                # },
                "suggestedStems": [],
            }
        ]

        if output_data.get("error_message"):
            request["answers"][0]["status"] = "error"
            request["answers"][0]["errorMessage"] = output_data.get("error_message")
        else:
            if output_data.get("context_song_info"):
                context_song_info_object = output_data.get("context_song_info")
                context_song_info = {}
                context_song_info["songId"] = context_song_info_object.get("songId")
                context_song_info["bpm"] = context_song_info_object.get("bpm")
                context_song_info["key"] = context_song_info_object.get("key")
                context_song_info["barCount"] = context_song_info_object.get("barCount")
                context_song_info["sectionName"] = context_song_info_object.get(
                    "sectionName"
                )
                context_song_info["sectionRole"] = context_song_info_object.get(
                    "sectionRole"
                )
                context_song_info["songStructure"] = context_song_info_object.get(
                    "songStructure"
                )
                context_song_info["contextAudioUris"] = context_song_info_object.get(
                    "contextAudioUris"
                )
                context_song_info["createdSectionsOrder"] = (
                    context_song_info_object.get("createdSectionsOrder")
                )
                context_song_info["arrangedSectionsOrder"] = (
                    context_song_info_object.get("arrangedSectionsOrder")
                )
                request["requestInformation"]["contextSongInfo"] = context_song_info

            request["requestInformation"]["workingSectionIndex"] = output_data.get(
                "working_section_index", 0
            )
            request["context"]["previousContext"] = output_data.get("previous_context")
            request["requestInformation"]["stemPrompts"] = output_data.get(
                "text_prompts"
            )
            request["requestInformation"]["globalMusicInformation"] = output_data.get(
                "target_music_info"
            )
            total_bar_count = 0
            if output_data.get("mix_stem_diff"):
                if not request["answers"][0].get("mix"):
                    request["answers"][0]["mix"] = {
                        "mixData": {
                            "stems": [],
                        },
                    }
                mix_id = str(uuid.uuid4())
                mix_data_object = {
                    "id": mix_id,
                    "dialogUuid": request["uuid"],
                    "status": "inProgress",
                }

                request["answers"][0]["mix"]["mixData"].update(mix_data_object)

                # Convert Stem objects to dictionaries if needed
                mix_stem_diff = output_data.get("mix_stem_diff")
                if mix_stem_diff:
                    converted_stems = []
                    for section in mix_stem_diff:
                        if isinstance(section, list):
                            converted_section = []
                            for stem in section:
                                if hasattr(stem, "to_dict"):  # Stem object
                                    converted_section.append(stem.to_dict())
                                else:  # Already a dictionary
                                    converted_section.append(stem)
                            converted_stems.append(converted_section)
                        else:
                            # Handle flat list structure
                            if hasattr(section, "to_dict"):
                                converted_stems.append(section.to_dict())
                            else:
                                converted_stems.append(section)
                    request["answers"][0]["mix"]["mixData"]["stems"] = converted_stems
                else:
                    request["answers"][0]["mix"]["mixData"]["stems"] = []

                stems_data = request["answers"][0]["mix"]["mixData"]["stems"]
                for idx_section in range(len(stems_data)):
                    for idx in range(len(stems_data[idx_section])):
                        stems_data[idx_section][idx]["mixId"] = mix_id
                    for idx_stem in range(len(stems_data[idx_section])):
                        if stems_data[idx_section][idx_stem].get("barCount", 0) > 0:
                            total_bar_count += stems_data[idx_section][idx_stem].get(
                                "barCount", 0
                            )
                            bpm = stems_data[idx_section][idx_stem].get("bpm", 0)
                            break

                request["answers"][0]["mix"]["mixData"][
                    "totalBarCount"
                ] = total_bar_count

                if total_bar_count:
                    request["answers"][0]["mix"]["mixData"]["totalDuration"] = round(
                        total_bar_count * 4 * 60 / bpm, 2
                    )

        return schema_data

    def _map_search_stems_from_es(
        self, output_data: Dict[str, Any], schema_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        request = schema_data["request"][0]

        if output_data.get("error_message"):
            request["answers"][0]["status"] = "error"
            request["answers"][0]["errorMessage"] = output_data.get("error_message")
        else:
            context_song_info_object = output_data.get("context_song_info")
            #  context_song_info의 키를 카멜케이스로 변환
            context_song_info = {}
            if context_song_info_object:
                context_song_info["songId"] = context_song_info_object.song_id
                context_song_info["bpm"] = context_song_info_object.bpm
                context_song_info["key"] = context_song_info_object.key
                context_song_info["barCount"] = context_song_info_object.bar_count
                context_song_info["sectionName"] = context_song_info_object.section_name
                context_song_info["sectionRole"] = context_song_info_object.section_role
                context_song_info["songStructure"] = (
                    context_song_info_object.song_structure
                )
                context_song_info["contextAudioUris"] = (
                    context_song_info_object.context_audio_uris
                )
                context_song_info["createdSectionsOrder"] = (
                    context_song_info_object.created_sections_order
                )
                context_song_info["arrangedSectionsOrder"] = (
                    context_song_info_object.arranged_sections_order
                )
                context_song_info["isRemix"] = context_song_info_object.is_remix

            request["requestInformation"]["contextSongInfo"] = context_song_info
            prompt_stem_info = output_data.get("prompt_stem_info")
            # answer_uuid = str(uuid.uuid4())
            for idx, stem_info in enumerate(prompt_stem_info):
                stem_object = {
                    "id": stem_info["id"],
                    "mixId": None,
                    "dialogUuid": request["answers"][0]["uuid"],
                    "isOriginal": False,
                    "isBlock": True,
                    "category": stem_info["stemType"],
                    "caption": stem_info["caption"],
                    "sectionName": context_song_info["sectionName"],
                    "sectionRole": context_song_info["sectionRole"],
                    "barCount": context_song_info["barCount"],
                    "bpm": context_song_info["bpm"],
                    "key": context_song_info["key"],
                }
                request["answers"][0]["suggestedStems"].append(stem_object)

            if output_data.get("mix_stem_diff"):
                if not request["answers"][0].get("mix"):
                    request["answers"][0]["mix"] = {
                        "mixData": {
                            "stems": [],
                        },
                    }
                mix_id = str(uuid.uuid4())
                mix_data_object = {
                    "id": mix_id,
                    "dialogUuid": request["uuid"],
                    "status": "inProgress",
                }

                request["answers"][0]["mix"]["mixData"].update(mix_data_object)
                request["answers"][0]["mix"]["mixData"]["stems"] = output_data.get(
                    "mix_stem_diff"
                )
        return schema_data

    def _map_generate_stem_diff(
        self, output_data: Dict[str, Any], schema_data: Dict[str, Any]
    ) -> Dict[str, Any]:

        request = schema_data["request"][0]
        if output_data.get("error_message"):
            request["answers"][0]["status"] = "error"
            request["answers"][0]["errorMessage"] = output_data.get("error_message")
        else:
            generated_uris = output_data.get("output_uris", [])

            idx_uris = 0
            # for idx_uris, uri in enumerate(generated_uris):
            #     request["answers"][idx]["stem"]["uri"] = uri
            for idx, answer in enumerate(request["answers"][0]["suggestedStems"]):
                answer["uri"] = generated_uris[idx_uris]
                idx_uris += 1

        return schema_data

    def _map_reply_orchestrator(
        self, output_data: Dict[str, Any], schema_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        request = schema_data["request"][0]
        if output_data.get("error_message"):
            request["answers"][0]["status"] = "error"
            request["answers"][0]["errorMessage"] = output_data.get("error_message")
        else:
            request["answers"][0]["status"] = "inProgress"
            request["answers"][0]["chatMessage"]["chatText"] = output_data.get("reply")

            if len(output_data.get("instrument_name")) != len(
                request["answers"][0]["suggestedStems"]
            ):
                print("⚠️ Warning: Instrument name of stem length mismatch.")
                # instrument_name_of_stem의 길이가 suggestedStems보다 짧을 때, 같은 값을 복사
                instrument_names = output_data.get("instrument_name", [])
                suggested_stems_count = len(request["answers"][0]["suggestedStems"])
                # suggestedStems의 길이만큼 instrument_name을 복사
                if instrument_names:
                    # 마지막 값을 복사하여 길이를 맞춤
                    last_instrument = instrument_names[-1]
                    while len(instrument_names) < suggested_stems_count:
                        instrument_names.append(last_instrument)

                # 복사된 instrument_names로 업데이트
                for idx, stem in enumerate(request["answers"][0]["suggestedStems"]):
                    if idx < len(instrument_names):
                        request["answers"][0]["suggestedStems"][idx][
                            "instrumentName"
                        ] = instrument_names[idx]

            else:
                for idx, stem in enumerate(request["answers"][0]["suggestedStems"]):
                    request["answers"][0]["suggestedStems"][idx]["instrumentName"] = (
                        output_data.get("instrument_name")[idx]
                    )

        return schema_data


class DataSchemaStore:
    """
    DataSchema 형태의 데이터를 저장하고 관리하는 클래스
    """

    def __init__(self, base_dir: str = "output"):
        """
        Args:
            base_dir: 데이터 저장 디렉토리
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.mapper = DataSchemaMapper()

    def get_schema_file_path(self, task_id: str) -> Path:
        """스키마 파일 경로 반환"""
        return self.base_dir / "data_schema" / f"data_schema_{task_id}.json"

    def load_or_create_schema(
        self, task_id: str, session_uuid: Optional[str] = None
    ) -> Dict[str, Any]:
        """기존 스키마 로드 또는 새로 생성"""
        schema_file = self.get_schema_file_path(task_id)

        if schema_file.exists():
            try:
                with open(schema_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"❌ Error loading schema file {schema_file}: {str(e)}")

        # 새로운 스키마 생성
        schema_data = self.mapper.create_base_schema(task_id, session_uuid)
        self.save_schema(task_id, schema_data)
        return schema_data

    def _default_encoder(self, obj):
        if obj.to_dict:
            return obj.to_dict()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def save_schema(self, task_id: str, schema_data: Dict[str, Any]) -> None:
        """스키마 데이터 저장"""
        try:
            schema_file = self.get_schema_file_path(task_id)

            # 타임스탬프 업데이트
            schema_data["request"][0]["updatedAt"] = datetime.now().isoformat()

            # answers 배열이 비어있지 않을 때만 업데이트
            if schema_data["request"][0].get("answers"):
                for answer in schema_data["request"][0]["answers"]:
                    if answer:
                        answer["updatedAt"] = datetime.now().isoformat()

            with open(schema_file, "w", encoding="utf-8") as f:
                json.dump(
                    schema_data,
                    f,
                    indent=2,
                    ensure_ascii=False,
                    default=self._default_encoder,
                )

            logger.info(f"Schema data saved to {schema_file}")

        except Exception as e:
            logger.error(f"❌ Error saving schema data: {str(e)}")

    def copy_schema_with_updated_info_for_publish_song(
        self,
        source_memory_id: str,
        new_task_id: str,
        user_prompt: str,
        intent_focused_prompt: str,
        reply_orchestrator_result: Dict[str, Any],
        session_uuid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        기존 메모리에서 대부분의 정보를 복사하고, 새로운 UUID, 시간, chatMessage, chatText, status만 업데이트
        """

        try:
            # 기존 스키마 데이터 로드
            source_task_id = (
                os.path.basename(source_memory_id)
                .replace(".json", "")
                .replace("data_schema_", "")
            )
            source_schema = self.load_or_create_schema(source_task_id)

            # 딥 카피 생성
            import copy

            new_schema = copy.deepcopy(source_schema)

            # 기존 mix에서 totalDuration 추출
            total_duration = None
            if (
                source_schema.get("request")
                and len(source_schema["request"]) > 0
                and source_schema["request"][0].get("answers")
                and len(source_schema["request"][0]["answers"]) > 0
            ):
                for answer in source_schema["request"][0]["answers"]:
                    if (
                        answer
                        and answer.get("mix")
                        and answer["mix"].get("mixData")
                        and answer["mix"]["mixData"].get("totalDuration")
                    ):
                        total_duration = answer["mix"]["mixData"]["totalDuration"]
                        break

            # 새로운 정보로 업데이트
            current_time = datetime.now().isoformat()

            # Request 레벨 업데이트
            request = new_schema["request"][0]
            request["uuid"] = new_task_id
            if session_uuid:
                request["sessionUuid"] = session_uuid
            request["createdAt"] = current_time
            request["updatedAt"] = current_time
            request["status"] = "finalized"

            # turnIndex 계산 - 이전 답변의 turnIndex에서 1 증가
            max_turn_index = request.get("turnIndex", 0)
            if request.get("answers") and len(request["answers"]) > 0:
                for answer in request["answers"]:
                    if answer and answer.get("turnIndex"):
                        max_turn_index = max(max_turn_index, answer["turnIndex"])
            new_turn_index = max_turn_index + 1

            # Request turnIndex 업데이트
            request["turnIndex"] = new_turn_index

            # ChatMessage 업데이트
            if not request.get("chatMessage"):
                request["chatMessage"] = {}

            request["chatMessage"].update(
                {
                    "uuid": str(uuid.uuid4()),
                    "dialogUuid": new_task_id,
                    "createdAt": current_time,
                    "chatText": user_prompt,
                    "intentFocusedPrompt": intent_focused_prompt,
                }
            )

            # requestInformation의 불필요한 정보들을 null로 설정 (publish_song 케이스에서는 새로운 생성이 아님)
            if request.get("requestInformation"):
                request["requestInformation"]["stemPrompts"] = None
                # contextSongInfo도 null로 설정 (새로운 섹션 정보가 필요하지 않음)
                for key in [
                    "songId",
                    "barCount",
                    "sectionName",
                    "sectionRole",
                    "songStructure",
                    "contextAudioUris",
                ]:
                    request["requestInformation"]["contextSongInfo"][key] = None

            # Answers 레벨 업데이트
            if request.get("answers") and len(request["answers"]) > 0:
                for answer in request["answers"]:
                    if answer:
                        answer["uuid"] = str(uuid.uuid4())
                        if session_uuid:
                            answer["sessionUuid"] = session_uuid
                        answer["createdAt"] = current_time
                        answer["updatedAt"] = current_time
                        answer["status"] = "finalized"
                        answer["requestUuid"] = new_task_id
                        answer["turnIndex"] = (
                            new_turn_index + 1
                        )  # 답변은 요청보다 1 더 큰 턴 인덱스

                        # suggestedStems를 null로 설정 (publish_song 케이스에서는 새로운 stem 제안이 아님)
                        answer["suggestedStems"] = None

                        # mix.mixData 정보 업데이트 (reply_orchestrator_result 정보 반영)
                        if answer.get("mix") and answer["mix"].get("mixData"):
                            mix_data = answer["mix"]["mixData"]
                            # 기존 totalDuration 유지 (있는 경우)
                            if total_duration is not None:
                                mix_data["totalDuration"] = total_duration

                            if reply_orchestrator_result.get("title"):
                                mix_data["title"] = reply_orchestrator_result["title"]
                            if reply_orchestrator_result.get("genre"):
                                mix_data["genre"] = reply_orchestrator_result["genre"]
                            if reply_orchestrator_result.get("mood"):
                                mix_data["mood"] = reply_orchestrator_result["mood"]
                            if reply_orchestrator_result.get("instruments"):
                                mix_data["instruments"] = reply_orchestrator_result[
                                    "instruments"
                                ]
                            if reply_orchestrator_result.get("music_caption"):
                                mix_data["musicCaption"] = reply_orchestrator_result[
                                    "music_caption"
                                ]
                            if reply_orchestrator_result.get("bpm"):
                                mix_data["bpm"] = reply_orchestrator_result["bpm"]
                            if reply_orchestrator_result.get("key"):
                                mix_data["key"] = reply_orchestrator_result["key"]
                            if reply_orchestrator_result.get("audio_url"):
                                mix_data["audioUrl"] = reply_orchestrator_result[
                                    "audio_url"
                                ]
                            if reply_orchestrator_result.get("thumbnail_url"):
                                mix_data["thumbnail"] = reply_orchestrator_result[
                                    "thumbnail_url"
                                ]

                        # Answer의 chatMessage 업데이트
                        if answer.get("chatMessage"):
                            answer["chatMessage"].update(
                                {
                                    "uuid": str(uuid.uuid4()),
                                    "dialogUuid": answer["uuid"],
                                    "createdAt": current_time,
                                    "chatText": reply_orchestrator_result["reply"],
                                }
                            )

            return new_schema

        except Exception as e:
            logger.error(f"❌ Error copying schema with updated info: {str(e)}")
            # 실패 시 기본 스키마 생성
            return self.create_base_schema(new_task_id, session_uuid)

    def update_tool_result(
        self,
        task_id: str,
        tool_name: str,
        output_data: Dict[str, Any],
        session_uuid: Optional[str] = None,
    ) -> None:
        """도구 결과로 스키마 업데이트"""

        try:
            schema_data = self.load_or_create_schema(task_id, session_uuid)

            # request에 실제 taskId 저장
            schema_data["request"][0]["taskId"] = task_id

            # 도구 결과 매핑
            updated_schema = self.mapper.map_tool_output(
                tool_name, output_data, schema_data
            )

            # 저장
            self.save_schema(task_id, updated_schema)

            logger.info(f"Updated schema for task {task_id} with {tool_name} result")

        except Exception as e:
            traceback.print_exc()
            logger.error(
                f"❌ Error updating schema with tool result [{tool_name}]: {str(e)}"
            )

    def list_schema_files(self) -> List[str]:
        """스키마 파일 목록 반환"""
        try:
            schema_files = []
            # Fix: our files are saved under output/data_schema/data_schema_*.json
            data_dir = self.base_dir / "data_schema"
            data_dir.mkdir(exist_ok=True)
            for file_path in data_dir.glob("data_schema_*.json"):
                schema_files.append(file_path.name)
            return sorted(schema_files)
        except Exception as e:
            logger.error(f"❌ Error listing schema files: {str(e)}")
            return []

    def get_most_recent_task_id(self) -> Optional[str]:
        """가장 최근에 생성된 task_id 반환 (error가 아닌 status만 고려)"""  # TODO: multiple handling for dataSchema
        try:
            schema_dir = self.base_dir / "data_schema"
            if not schema_dir.exists():
                logger.warning(f"Schema directory {schema_dir} does not exist")
                return None

            # data_schema_*.json 파일들을 찾아서 처리
            schema_files = []
            for file_path in schema_dir.glob("data_schema_*.json"):
                if file_path.is_file():
                    try:
                        # JSON 파일 읽기
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                            # request 배열에서 status가 error가 아닌 항목 찾기
                        valid_requests = []
                        if "request" in data and isinstance(data["request"], list):
                            for request in data["request"]:
                                if (
                                    isinstance(request, dict)
                                    and request.get("status") != "error"
                                ):
                                    valid_requests.append(request)

                        # 유효한 request가 있는 경우만 처리
                        if valid_requests:
                            # 파일명에서 task_id 추출
                            task_id = file_path.stem.replace("data_schema_", "")
                            # 파일 생성 시간 가져오기
                            creation_time = file_path.stat().st_ctime
                            schema_files.append((task_id, creation_time, file_path))
                            logger.debug(
                                f"Valid task_id found: {task_id} with {len(valid_requests)} non-error requests"
                            )
                        else:
                            logger.debug(
                                f"Skipping {file_path.name}: no valid (non-error) requests found"
                            )

                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        logger.warning(
                            f"❌ Error reading schema file {file_path.name}: {str(e)}"
                        )
                        continue

            if not schema_files:
                logger.warning("No schema files with valid (non-error) requests found")
                return None

            # 생성 시간 기준으로 정렬 (최신이 맨 위)
            schema_files.sort(key=lambda x: x[1], reverse=True)

            most_recent_task_id = schema_files[0][0]
            logger.info(
                f"Most recent task_id with valid requests: {most_recent_task_id}"
            )
            return most_recent_task_id

        except Exception as e:
            logger.error(f"❌ Error finding most recent task_id: {str(e)}")
            return None

    def load_most_recent_schema(self) -> Optional[Dict[str, Any]]:
        """가장 최근에 생성된 스키마 파일 로드"""
        try:
            most_recent_task_id = self.get_most_recent_task_id()
            if most_recent_task_id is None:
                return None

            return self.load_or_create_schema(most_recent_task_id)

        except Exception as e:
            logger.error(f"❌ Error loading most recent schema: {str(e)}")
            return None

    def extract_context_from_schema(
        self, schema_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """스키마 데이터에서 context 정보 추출"""
        try:
            if (
                not schema_data
                or "request" not in schema_data
                or not schema_data["request"]
            ):
                return {}

            request = schema_data["request"][0]
            request_information = request.get("requestInformation", {})
            answers = request.get("answers", [])
            # context_song_info 추출
            context_song_info = request_information.get("contextSongInfo", {})
            # TODO ? check formart
            # context_song_info = self.mapper._convert_dict_keys_to_camel_case(
            #     context_song_info
            # )

            working_section_index = request_information.get("workingSectionIndex", 0)

            if answers[0].get("mix"):
                mix_stem_diff = answers[0]["mix"]["mixData"].get("stems", [])
            else:
                mix_stem_diff = []
            return {
                "context_song_info": context_song_info,
                "working_section_index": working_section_index,
                "previous_context": request["context"].get("previousContext", []),
                "turn_index": answers[0]["turnIndex"],
                "mix_stem_diff": mix_stem_diff,
                "generated_stem_diff": answers[0]["suggestedStems"],
                "task_id": request.get("taskId"),
                "session_uuid": request.get("sessionUuid"),
            }

        except Exception as e:
            logger.error(f"❌ Error extracting context from schema: {str(e)}")
            return {}


# ===== Global Store Instance =====

_global_data_schema_store = None


def get_data_schema_store() -> DataSchemaStore:
    """
    전역 DataSchemaStore 인스턴스 반환 (싱글톤 패턴)
    """
    global _global_data_schema_store
    if _global_data_schema_store is None:
        _global_data_schema_store = DataSchemaStore()
    return _global_data_schema_store
