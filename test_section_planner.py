#!/usr/bin/env python3
"""
get_next_section_info_llm 함수 테스트 스크립트
"""

import asyncio
import os
import sys

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.get_stem_section_info import get_next_section_info_llm


async def test_section_planner():
    """섹션 플래너 함수 테스트"""

    # 예시 데이터 (제공된 JSON 데이터를 기반으로)
    context_song_info = {
        "songId": "b003309_attention",
        "bpm": 88,
        "key": "Dm",
        "barCount": 4,
        "sectionName": "H",
        "sectionRole": "Chorus / Drop",
        "songStructure": {
            "A": "Verse",
            "B": "Verse",
            "C": "Verse",
            "D": "Verse",
            "E": "Verse",
            "F": "Verse",
            "G": "Chorus / Drop",
            "H": "Chorus / Drop",
            "I": "Verse",
            "J": "Verse",
            "K": "Verse",
            "L": "Chorus / Drop",
            "M": "Chorus / Drop",
            "N": "Chorus / Drop",
            "O": "Outro",
            "P": "Outro",
        },
        "createdSectionsOrder": [
            {"I": "Verse"},
            {"L": "Chorus / Drop"},
            {"M": "Chorus / Drop"},
            {"O": "Outro"},
            {"P": "Outro"},
            {"H": "Chorus / Drop"},
        ],
        "arrangedSectionsOrder": [
            {"H": "Chorus / Drop"},
            {"I": "Verse"},
            {"L": "Chorus / Drop"},
            {"M": "Chorus / Drop"},
            {"O": "Outro"},
            {"P": "Outro"},
        ],
        "workingSectionIndex": 0,
    }

    mix_stem_diff = [
        {"category": "rhythm", "caption": "드럼과 퍼커션"},
        {"category": "low", "caption": "베이스 라인"},
        {"category": "mid", "caption": "피아노와 신디사이저"},
        {"category": "high", "caption": "멜로디와 보컬"},
    ]

    print("=== 섹션 플래너 테스트 ===")
    print(
        f"현재 섹션: {context_song_info['sectionName']} ({context_song_info['sectionRole']})"
    )
    print(f"생성된 섹션 순서: {context_song_info['createdSectionsOrder']}")
    print(f"배열된 섹션 순서: {context_song_info['arrangedSectionsOrder']}")
    print(f"현재 믹스 스템: {[stem['category'] for stem in mix_stem_diff]}")
    print()

    # 테스트 1: 자동 진행 모드
    # print("=== 테스트 1: 자동 진행 모드 ===")
    # try:
    #     result = get_next_section_info_llm(context_song_info, mix_stem_diff)
    #     print("결과:")
    #     for key, value in result.items():
    #         print(f"  {key}: {value}")
    #     print()
    # except Exception as e:
    #     print(f"오류 발생: {e}")
    #     print()

    # 테스트 2: F 섹션 추가 후 다음 섹션 예측
    # print("=== 테스트 2: F 섹션 추가 후 다음 단계 ===")
    # # F가 추가된 상황을 시뮬레이션
    # context_song_info_step2 = context_song_info.copy()
    # context_song_info_step2["sectionName"] = "F"
    # context_song_info_step2["sectionRole"] = "Verse"
    # context_song_info_step2["created_sections_order"] = [
    #     ("I", "Verse"),
    #     ("L", "Chorus / Drop"),
    #     ("M", "Chorus / Drop"),
    #     ("O", "Outro"),
    #     ("P", "Outro"),
    #     ("H", "Chorus / Drop"),
    #     ("F", "Verse"),
    # ]
    # context_song_info_step2["arranged_sections_order"] = [
    #     ("F", "Verse"),
    #     ("H", "Chorus / Drop"),
    #     ("I", "Verse"),
    #     ("L", "Chorus / Drop"),
    #     ("M", "Chorus / Drop"),
    #     ("O", "Outro"),
    #     ("P", "Outro"),
    # ]

    # try:
    #     result = get_next_section_info_llm(context_song_info_step2, mix_stem_diff)
    #     print("G 섹션 추가 후 상황:")
    #     print(
    #         f"  현재 섹션: {context_song_info_step2['sectionName']} ({context_song_info_step2['sectionRole']})"
    #     )
    #     print(f"  생성된 섹션: {context_song_info_step2['created_sections_order']}")
    #     print(f"  배열된 섹션: {context_song_info_step2['arranged_sections_order']}")
    #     print("다음 섹션 예측 결과:")
    #     for key, value in result.items():
    #         print(f"  {key}: {value}")
    #     print()
    # except Exception as e:
    #     print(f"오류 발생: {e}")
    #     print()

    # 테스트 3: 사용자 특정 요청
    print("=== 테스트 3: 사용자 특정 요청 ===")
    user_request = "방금 생성한 섹션 뒤에 코러스 섹션을 만들고 싶어요"
    try:
        result = await get_next_section_info_llm(
            context_song_info, mix_stem_diff, user_request
        )
        print(f"사용자 요청: {user_request}")
        print("결과:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        print()
    except Exception as e:
        print(f"오류 발생: {e}")
        print()


if __name__ == "__main__":
    asyncio.run(test_section_planner())
