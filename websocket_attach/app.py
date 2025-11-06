import json
import os
import logging
from common.cache import cache_set
from common.websocket import send_websocket_message

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    print(f"Event: {event}")

    message = json.loads(event["body"])
    request_uuid = message["requestUuid"]
    cache_set(f"connection_id:{request_uuid}", event["requestContext"]["connectionId"])

    send_websocket_message(
        event["requestContext"]["connectionId"],
        {"systemMessage": f"requestUuid:{request_uuid} mapped to this connection"},
    )

    return {
        "statusCode": 200,
        "body": f"requestUuid:{request_uuid} mapped to this connection",
    }

    return {"statusCode": 200, "body": "Message saved"}

    websocket_api_endpoint = os.environ.get("WEBSOCKET_API_ENDPOINT").replace(
        "wss://", "https://"
    )  # Assuming the endpoint is stored in the function's environment variables

    import time

    time.sleep(10)

    message = {
        "status": "completed",
        "logs": [],
        "data": {
            "global_music_information": {
                "genre": {"primary": "lofi", "secondary": ["jazz", "hip hop"]},
                "mood": ["comfortable", "relaxing", "chill", "mellow", "peaceful"],
                "theme": ["relaxation", "studying", "ambient", "background music"],
                "bpm": {"min": 70, "max": 90},
                "scale": "both",
                "instrumentation": [
                    "electric piano",
                    "jazz guitar",
                    "upright bass",
                    "soft drums",
                    "vinyl crackle",
                    "saxophone",
                    "trumpet",
                    "synthesizer pads",
                ],
                "stem_selection": ["mid", "rhythm", "low", "high", "fx"],
                "stem_selection_reasoning": "This is a complete track generation request for lofi jazz hip hop music. Since the user wants to \"create\" entirely new tracks from scratch, all stems are needed: 'mid' for the core harmonic elements like electric piano and jazz chords; 'rhythm' for the characteristic laid-back hip hop drum patterns; 'low' for the bass foundation typical in jazz and hip hop; 'high' for melodic instruments like saxophone or guitar leads that define the jazz character; and 'fx' for the essential lofi textures like vinyl crackle and atmospheric elements that create the comfortable, relaxing ambiance.",
                "reasoning": "Lofi jazz hip hop requires a specific combination of relaxed hip hop rhythms (70-90 BPM), jazz harmonic progressions, and lo-fi production aesthetics. The comfortable and relaxing qualities call for warm, mellow instrumentation with characteristic vinyl textures. Both major and minor scales work well in this genre, with minor often used for moodier sections and major for brighter, more uplifting moments. The instrumentation focuses on classic jazz elements (piano, bass, horns) combined with hip hop production techniques and lofi processing effects.",
                "request_type": "generate",
                "intent_focused_prompt": "Create a comfortable and relaxing lofi jazz hiphop music track",
            },
            "mix_info": [
                [
                    {
                        "id": "b000981_better-f-mid",
                        "stemType": "mid",
                        "sectionName": "F",
                        "key": "CM",
                        "bpm": 80,
                        "barCount": 4,
                        "caption": "[0~10s] The electric piano delivers warm, smooth chords with a mellow, rounded tone that effortlessly grounds the harmony. Its gentle, flowing voicings create a lush, intimate atmosphere, enhancing the track's relaxing, groovy vibe. Meanwhile, the electric guitar adds subtle, shimmering textures with clean, crisp strums and light, expressive licks, weaving melodic accents that enrich the soundscape without overpowering. Together, these instruments blend to form a rich, dynamic foundation that perfectly complements the lo-fi and alternative R&B elements. Their interplay fosters a joyful yet chill mood, inviting listeners into a soulful",
                        "gain": 1,
                        "targetBpm": 80,
                        "targetKey": "CM",
                        "keyDifference": 0,
                        "barOffset": [0],
                        "url": "https://ai-agent-data-new.s3.ap-northeast-2.amazonaws.com/block_data/b000981_better/F/mid/b000981_better-f-mid.aac",
                    },
                    {
                        "id": "b003765_lovelyplace-e-rhythm",
                        "stemType": "rhythm",
                        "sectionName": "E",
                        "key": "F#M",
                        "bpm": 91,
                        "barCount": 8,
                        "caption": "[0~10s] This drum track layers crisp claps, shimmering cymbals, tight hi-hats, and a warm, punchy kick to create a smooth, laid-back groove. The claps offer a sharp, rhythmic snap that accentuates the beat, while the hi-hat weaves delicate, fluttering patterns adding subtle momentum. Cymbals add bright, airy textures that lift the mix without overpowering, and the kick grounds the rhythm with a rounded low-end presence. Together, these elements craft a chilled yet engaging pulse, perfectly complementing the song’s trap soul style. Their inter [10~20s] This drum track layers crisp claps, shimmering cymbals, tight hi-hats, and a warm, punchy kick to create a smooth, laid-back groove. The claps offer a sharp, rhythmic snap that accentuates the beat, while the hi-hat weaves delicate, fluttering patterns adding subtle momentum. Cymbals add bright, airy textures that lift the mix without overpowering, and the kick grounds the rhythm with a rounded low-end presence. Together, these elements craft a chilled yet engaging pulse, perfectly complementing the song’s trap soul style. Their inter",
                        "gain": 1,
                        "targetKey": "CM",
                        "keyDifference": 0,
                        "targetBpm": 80,
                        "barOffset": [-4],
                        "url": "https://ai-agent-data-new.s3.ap-northeast-2.amazonaws.com/block_data/b003765_lovelyplace/E/rhythm/b003765_lovelyplace-e-rhythm.aac",
                    },
                    {
                        "id": "p000235_image-e-low",
                        "stemType": "low",
                        "sectionName": "E",
                        "key": "A#M",
                        "bpm": 85,
                        "barCount": 8,
                        "caption": "[0~10s] The bass guitar delivers a warm, rounded tone with gentle articulation, anchoring the groove with smooth, mellow rhythm that subtly pulses beneath the mix. Its soft, low-frequency presence creates a rich foundation, weaving through the track with understated melodic movement that enhances the dreamy and relaxing atmosphere. By maintaining a laid-back, consistent groove, the bass supports the chill, lo-fi vibe while adding a touch of soulful warmth. This instrument’s mellow depth and gentle groove foster a peaceful, cozy soundscape, perfectly complementing the calm and groovy essence of the overall hip-hop [10~20s] The bass guitar delivers a warm, rounded tone with gentle articulation, anchoring the groove with smooth, mellow rhythm that subtly pulses beneath the mix. Its soft, low-frequency presence creates a rich foundation, weaving through the track with understated melodic movement that enhances the dreamy and relaxing atmosphere. By maintaining a laid-back, consistent groove, the bass supports the chill, lo-fi vibe while adding a touch of soulful warmth. This instrument’s mellow depth and gentle groove foster a peaceful, cozy soundscape, perfectly complementing the calm and groovy essence of the overall hip-hop",
                        "gain": 1,
                        "targetKey": "CM",
                        "keyDifference": 2,
                        "targetBpm": 80,
                        "barOffset": [-4],
                        "url": "https://ai-agent-data-new.s3.ap-northeast-2.amazonaws.com/block_data/p000235_image/E/low/p000235_image-e-low.aac",
                    },
                    {
                        "id": "p000873_august8pm-d-high",
                        "stemType": "high",
                        "sectionName": "D",
                        "key": "GM",
                        "bpm": 72,
                        "barCount": 8,
                        "caption": "[0~10s] The clean electric guitar features a smooth, shimmering tone with gentle articulation and subtle reverb, creating an intimate, spacious sound. Its crisp, articulate notes weave melodic lines that delicately float above the mix, offering warmth without overpowering other elements. The instrument’s restrained phrasing and spacious resonance evoke a sense of wonder and tranquility, perfectly complementing the slow, touching mood of the song. By avoiding aggressive distortion, the guitar maintains a natural, organic presence that invites listeners into a reflective, heartfelt space. Its understated articulation enhances the pop sensibility, grounding the track while deepening the emotional [10~20s] The clean electric guitar features a smooth, shimmering tone with gentle, sustained notes that weave delicate melodic lines throughout the mix. Its crisp articulation and subtle reverb create an intimate, spacious atmosphere, avoiding heavy distortion to maintain clarity and warmth. The instrument’s slow, measured strumming provides a soothing harmonic foundation, allowing each note to resonate with clarity and emotional depth. By layering spacious, airy chords, the guitar enhances the track's touching and peaceful quality, inviting listeners into a reflective, heartfelt soundscape. Its serene timbre perfectly complements the singer-songwriter style, emphasizing vulnerability",
                        "gain": 1,
                        "targetKey": "CM",
                        "keyDifference": 5,
                        "targetBpm": 80,
                        "barOffset": [-4],
                        "url": "https://ai-agent-data-new.s3.ap-northeast-2.amazonaws.com/block_data/p000873_august8pm/D/high/p000873_august8pm-d-high.aac",
                    },
                    {
                        "id": "b002481_ever-f-fx",
                        "stemType": "fx",
                        "sectionName": "F",
                        "key": "Fm",
                        "bpm": 90,
                        "barCount": 8,
                        "caption": "[0~10s] The ambience and atmospheric layers create a lush, immersive soundscape that gently envelops the mix with warm, textured tones. Subtle, evolving pads and soft background textures add depth and softness, enhancing the track’s dreamy and soulful vibe. These sonic elements provide a spacious foundation, allowing the lo-fi Hip-Hop and Neo-Soul rhythms to breathe naturally. Their smooth, warm presence enriches the emotional warmth and passion of the song, while their understated, groovy texture complements the chill, mellow tempo. Overall, the ambience subtly intensifies the dynamic flow, height [10~20s] The ambience and atmospheric layers create a lush, immersive soundscape that gently envelops the mix with warm, textured tones. Subtle, evolving pads and soft background textures add depth and softness, enhancing the track’s dreamy and soulful vibe. These sonic elements provide a spacious foundation, allowing the lo-fi Hip-Hop and Neo-Soul rhythms to breathe naturally. Their smooth, warm presence enriches the emotional warmth and passion of the song, while their understated, groovy texture complements the chill, mellow tempo. Overall, the ambience subtly intensifies the dynamic flow, height",
                        "gain": 1,
                        "targetKey": "CM",
                        "keyDifference": 0,
                        "targetBpm": 80,
                        "barOffset": [-4],
                        "url": "https://ai-agent-data-new.s3.ap-northeast-2.amazonaws.com/block_data/b002481_ever/F/fx/b002481_ever-f-fx.aac",
                    },
                ]
            ],
        },
    }

    response = {"statusCode": 200, "body": "Message sent"}

    try:
        # Assuming boto3 is available and configured
        import boto3

        apigateway_management_api = boto3.client(
            "apigatewaymanagementapi", endpoint_url=websocket_api_endpoint
        )

        apigateway_management_api.post_to_connection(
            Data=json.dumps(message), ConnectionId=connection_id
        )
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        response = {"statusCode": 500, "body": "Failed to send message"}

    return response
