#!/usr/bin/env python3
"""
Test script for AI Agent Stem Diff API
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("🏥 Health Check:", response.json())
    return response.status_code == 200

def test_root():
    """Test root endpoint"""
    response = requests.get(f"{BASE_URL}/")
    print("🏠 Root:", response.json())
    return response.status_code == 200

def test_generate_stem_diff_simple():
    """Test simple stem diff generation"""
    payload = {
        "text_prompts": [
            {
                "category": "mixed",
                "text": "heavy rock music, heavy metal guitar, heavy metal drums",
                "uri": ""
            },
            {
                "category": "rhythm", 
                "text": "heavy rock drums", 
                "uri": ""
            }
        ],
        "context_song_info": {}
    }
    
    response = requests.post(f"{BASE_URL}/generate-stem-diff-simple", json=payload)
    print("🎵 Simple Stem Diff Response:", response.json())
    return response.status_code == 200

def test_generate_stem_diff_full():
    """Test full stem diff generation with proper request model"""
    payload = {
        "text_prompts": {
            "text_prompts": [
                {
                    "category": "low",
                    "text": "synth bass, rock style bass",
                    "uri": ""
                }
            ]
        },
        "context_song_info": {
            "song_id": "b005525_fallapart",
            "bpm": 120,
            "key": "GM",
            "bar_count": 4,
            "section_name": "N",
            "section_role": "Chorus / Drop",
            "context_audio_uris": [
                "s3://ai-agent-data-new/block_data/b005525_fallapart/N/high/b005525_fallapart-n-high",
                "s3://ai-agent-data-new/block_data/b005525_fallapart/N/low/b005525_fallapart-n-low",
                "s3://ai-agent-data-new/block_data/b005525_fallapart/N/mid/b005525_fallapart-n-mid",
                "s3://ai-agent-data-new/block_data/b005525_fallapart/N/rhythm/b005525_fallapart-n-rhythm",
            ],
            "generated_mix_uris": [
                "s3://ai-agent-data-new/generated_stem/1f4896ec-f54c-47c3-b792-ee20085c23b2-rhythm",
            ]
        }
    }
    
    response = requests.post(f"{BASE_URL}/generate-stem-diff", json=payload)
    print("🎵 Full Stem Diff Response:", response.json())
    return response.status_code == 200

def test_example_endpoints():
    """Test example endpoints"""
    # Test example 1
    response1 = requests.post(f"{BASE_URL}/generate-stem-diff/example-1")
    print("🎵 Example 1 Response:", response1.json())
    
    # Test example 2
    response2 = requests.post(f"{BASE_URL}/generate-stem-diff/example-2")
    print("🎵 Example 2 Response:", response2.json())
    
    return response1.status_code == 200 and response2.status_code == 200

def main():
    """Run all tests"""
    print("🧪 Starting API Tests...")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health),
        ("Root Endpoint", test_root),
        ("Simple Stem Diff", test_generate_stem_diff_simple),
        ("Full Stem Diff", test_generate_stem_diff_full),
        ("Example Endpoints", test_example_endpoints),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"✅ {test_name}: {'PASS' if success else 'FAIL'}")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\n🎯 Overall: {passed}/{total} tests passed")

if __name__ == "__main__":
    main() 