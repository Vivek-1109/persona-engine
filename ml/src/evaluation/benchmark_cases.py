"""
Comprehensive benchmark suite of 65 synthetic and derived conversation cases for Stage 6E.
Covers all 16 benchmark categories across 1, 3, 5, 7, and 11 turn dialogues.
Strictly adheres to privacy rules: no private raw WhatsApp logs or personal credentials.
"""

from typing import List
from ml.src.evaluation.evaluation_schema import BenchmarkCase, BenchmarkCategory


# Standard synthetic memory bank for testing retrieval and contamination
STANDARD_MEMORY_BANK = [
    {
        "id": "mem_gaming_pref",
        "memory_type": "preference",
        "content": "Likes gaming",
        "topic": "gaming",
        "importance": "high",
    },
    {
        "id": "mem_gaming_exp",
        "memory_type": "experience",
        "content": "Participated in a gaming tournament",
        "topic": "gaming",
        "importance": "medium",
    },
    {
        "id": "mem_college_fact",
        "memory_type": "fact",
        "content": "Studies B.Tech CSE",
        "topic": "college",
        "importance": "high",
    },
    {
        "id": "mem_college_loc",
        "memory_type": "fact",
        "content": "College is located in Noida",
        "topic": "college",
        "importance": "medium",
    },
    {
        "id": "mem_tech_laptop",
        "memory_type": "fact",
        "content": "Owns an Acer Nitro laptop",
        "topic": "technology",
        "importance": "medium",
    },
    {
        "id": "mem_tech_java",
        "memory_type": "preference",
        "content": "Prefers Java",
        "topic": "technology",
        "importance": "high",
    },
    {
        "id": "mem_movie_marvel",
        "memory_type": "preference",
        "content": "Likes Marvel movies",
        "topic": "movies",
        "importance": "high",
    },
    {
        "id": "mem_social_friend",
        "memory_type": "relationship",
        "content": "Rahul is a college friend",
        "topic": "college",
        "importance": "high",
    },
    {
        "id": "mem_plan_interview",
        "memory_type": "plan",
        "content": "Has an upcoming interview",
        "topic": "college",
        "importance": "low",
    },
]


BENCHMARK_CASES: List[BenchmarkCase] = [
    # =========================================================================
    # A. CASUAL CONVERSATION (4 cases)
    # =========================================================================
    BenchmarkCase(
        case_id="CASE_CASUAL_01",
        category=BenchmarkCategory.CASUAL_CONVERSATION,
        conversation=[
            {"role": "user", "content": "Kaisa hai bhai?"}
        ],
        expected_topic="casual_chat",
        expected_intent="question",
        expected_strategy="answer",
        expected_context_behavior="Opening inquiry in casual context",
        expected_use_memory=False,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Standard casual greeting turn.",
    ),
    BenchmarkCase(
        case_id="CASE_CASUAL_02",
        category=BenchmarkCategory.CASUAL_CONVERSATION,
        conversation=[
            {"role": "user", "content": "kya chal raha hai aajkal?"}
        ],
        expected_topic="casual_chat",
        expected_intent="question",
        expected_strategy="answer",
        expected_context_behavior="General status query",
        expected_use_memory=False,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Broad status check.",
    ),
    BenchmarkCase(
        case_id="CASE_CASUAL_03",
        category=BenchmarkCategory.CASUAL_CONVERSATION,
        conversation=[
            {"role": "assistant", "content": "aur bata kaisa hai"},
            {"role": "user", "content": "bas badiya chal raha"}
        ],
        expected_topic="casual_chat",
        expected_intent="casual_chat",
        expected_strategy="react",
        expected_context_behavior="Ongoing casual chit-chat",
        expected_use_memory=False,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Positive casual update.",
    ),
    BenchmarkCase(
        case_id="CASE_CASUAL_04",
        category=BenchmarkCategory.CASUAL_CONVERSATION,
        conversation=[
            {"role": "user", "content": "sach me yaar"}
        ],
        expected_topic="casual_chat",
        expected_intent="reaction",
        expected_strategy="react",
        expected_context_behavior="Short reactive utterance",
        expected_use_memory=False,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Brief conversational affirmation.",
    ),

    # =========================================================================
    # B. GAMING (4 cases)
    # =========================================================================
    BenchmarkCase(
        case_id="CASE_GAMING_01",
        category=BenchmarkCategory.GAMING,
        conversation=[
            {"role": "user", "content": "bhai bgmi khelega?"}
        ],
        expected_topic="gaming",
        expected_intent="invitation",
        expected_strategy="accept",
        expected_context_behavior="Direct gaming invitation",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Gaming invitation should select gaming memory and accept.",
    ),
    BenchmarkCase(
        case_id="CASE_GAMING_02",
        category=BenchmarkCategory.GAMING,
        conversation=[
            {"role": "assistant", "content": "match jeet gaye kal"},
            {"role": "user", "content": "aaj fir valorant chalte hain"}
        ],
        expected_topic="gaming",
        expected_intent="invitation",
        expected_strategy="accept",
        expected_context_behavior="Valorant gaming proposal",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="PC gaming invitation.",
    ),
    BenchmarkCase(
        case_id="CASE_GAMING_03",
        category=BenchmarkCategory.GAMING,
        conversation=[
            {"role": "user", "content": "rank push karna hai squad ke saath"}
        ],
        expected_topic="gaming",
        expected_intent="information",
        expected_strategy="react",
        expected_context_behavior="Gaming rank push statement",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Gaming progress discussion.",
    ),
    BenchmarkCase(
        case_id="CASE_GAMING_04",
        category=BenchmarkCategory.GAMING,
        conversation=[
            {"role": "assistant", "content": "khelte hain sham ko"},
            {"role": "user", "content": "steam pe download ho gaya game"}
        ],
        expected_topic="gaming",
        expected_intent="information",
        expected_strategy="react",
        expected_context_behavior="Steam download complete",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Gaming installation confirmation.",
    ),

    # =========================================================================
    # C. COLLEGE (4 cases)
    # =========================================================================
    BenchmarkCase(
        case_id="CASE_COLLEGE_01",
        category=BenchmarkCategory.COLLEGE,
        conversation=[
            {"role": "user", "content": "Dean ne attendance ka notice nikala hai"}
        ],
        expected_topic="college",
        expected_intent="information",
        expected_strategy="provide_information",
        expected_context_behavior="Academic attendance warning",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Academic notice requires serious tone.",
    ),
    BenchmarkCase(
        case_id="CASE_COLLEGE_02",
        category=BenchmarkCategory.COLLEGE,
        conversation=[
            {"role": "user", "content": "semester exam date sheet aa gayi kya?"}
        ],
        expected_topic="college",
        expected_intent="question",
        expected_strategy="answer",
        expected_context_behavior="Exam schedule query",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Examination timetable question.",
    ),
    BenchmarkCase(
        case_id="CASE_COLLEGE_03",
        category=BenchmarkCategory.COLLEGE,
        conversation=[
            {"role": "assistant", "content": "lab report submit ki?"},
            {"role": "user", "content": "haan sir ko de di assignment"}
        ],
        expected_topic="college",
        expected_intent="information",
        expected_strategy="provide_information",
        expected_context_behavior="Assignment submission update",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Academic coursework status.",
    ),
    BenchmarkCase(
        case_id="CASE_COLLEGE_04",
        category=BenchmarkCategory.COLLEGE,
        conversation=[
            {"role": "user", "content": "kal 9 baje class hai attendance mandatory hai"}
        ],
        expected_topic="college",
        expected_intent="information",
        expected_strategy="provide_information",
        expected_context_behavior="Morning lecture attendance notice",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Compulsory lecture alert.",
    ),

    # =========================================================================
    # D. TECHNOLOGY (4 cases)
    # =========================================================================
    BenchmarkCase(
        case_id="CASE_TECH_01",
        category=BenchmarkCategory.TECHNOLOGY,
        conversation=[
            {"role": "user", "content": "bhai battery bahut jaldi drain ho rahi"}
        ],
        expected_topic="technology",
        expected_intent="information",
        expected_strategy="suggest",
        expected_context_behavior="Hardware battery defect report",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Technical problem report warrants supportive suggestion.",
    ),
    BenchmarkCase(
        case_id="CASE_TECH_02",
        category=BenchmarkCategory.TECHNOLOGY,
        conversation=[
            {"role": "user", "content": "code me null pointer error aa raha hai"}
        ],
        expected_topic="technology",
        expected_intent="information",
        expected_strategy="suggest",
        expected_context_behavior="Software runtime exception",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Coding debugging problem.",
    ),
    BenchmarkCase(
        case_id="CASE_TECH_03",
        category=BenchmarkCategory.TECHNOLOGY,
        conversation=[
            {"role": "assistant", "content": "backend kispe bana raha hai?"},
            {"role": "user", "content": "Spring Boot or Django me se kaun sa best hai?"}
        ],
        expected_topic="technology",
        expected_intent="question",
        expected_strategy="answer",
        expected_context_behavior="Backend stack architectural question",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Programming framework consultation.",
    ),
    BenchmarkCase(
        case_id="CASE_TECH_04",
        category=BenchmarkCategory.TECHNOLOGY,
        conversation=[
            {"role": "user", "content": "laptop crash ho raha bar bar"}
        ],
        expected_topic="technology",
        expected_intent="information",
        expected_strategy="suggest",
        expected_context_behavior="Operating system / hardware crash",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Hardware instability symptom.",
    ),

    # =========================================================================
    # E. MOVIES (4 cases)
    # =========================================================================
    BenchmarkCase(
        case_id="CASE_MOVIES_01",
        category=BenchmarkCategory.MOVIES,
        conversation=[
            {"role": "user", "content": "Marvel ki new movie dekhi kya?"}
        ],
        expected_topic="movies",
        expected_intent="question",
        expected_strategy="answer",
        expected_context_behavior="Cinema discussion focusing on Marvel",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Marvel movie query leverages movie preference.",
    ),
    BenchmarkCase(
        case_id="CASE_MOVIES_02",
        category=BenchmarkCategory.MOVIES,
        conversation=[
            {"role": "user", "content": "weekend pe cinema chalte hain new film dekhne"}
        ],
        expected_topic="movies",
        expected_intent="invitation",
        expected_strategy="accept",
        expected_context_behavior="Movie outing proposal",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Film outing invitation.",
    ),
    BenchmarkCase(
        case_id="CASE_MOVIES_03",
        category=BenchmarkCategory.MOVIES,
        conversation=[
            {"role": "assistant", "content": "trailer kaisa laga?"},
            {"role": "user", "content": "visuals bahut zabardast the movie ke"}
        ],
        expected_topic="movies",
        expected_intent="information",
        expected_strategy="react",
        expected_context_behavior="Film review commentary",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Cinematic reaction comment.",
    ),
    BenchmarkCase(
        case_id="CASE_MOVIES_04",
        category=BenchmarkCategory.MOVIES,
        conversation=[
            {"role": "user", "content": "Avengers ka review dekha kya?"}
        ],
        expected_topic="movies",
        expected_intent="question",
        expected_strategy="answer",
        expected_context_behavior="Film critique inquiry",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Franchise movie critique question.",
    ),

    # =========================================================================
    # F. PLANNING (4 cases)
    # =========================================================================
    BenchmarkCase(
        case_id="CASE_PLAN_01",
        category=BenchmarkCategory.PLANNING,
        conversation=[
            {"role": "user", "content": "kal kab milte hain?"}
        ],
        expected_topic="plans",
        expected_intent="planning",
        expected_strategy="suggest",
        expected_context_behavior="Meeting time coordination",
        expected_use_memory=False,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Time planning turn.",
    ),
    BenchmarkCase(
        case_id="CASE_PLAN_02",
        category=BenchmarkCategory.PLANNING,
        conversation=[
            {"role": "user", "content": "weekend ka kya plan hai?"}
        ],
        expected_topic="plans",
        expected_intent="planning",
        expected_strategy="suggest",
        expected_context_behavior="Weekend schedule query",
        expected_use_memory=False,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Broad weekend planning.",
    ),
    BenchmarkCase(
        case_id="CASE_PLAN_03",
        category=BenchmarkCategory.PLANNING,
        conversation=[
            {"role": "assistant", "content": "canteen kab chale?"},
            {"role": "user", "content": "lunch break me chalte hain 1 baje"}
        ],
        expected_topic="plans",
        expected_intent="planning",
        expected_strategy="suggest",
        expected_context_behavior="Lunch scheduling",
        expected_use_memory=False,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Midday break proposal.",
    ),
    BenchmarkCase(
        case_id="CASE_PLAN_04",
        category=BenchmarkCategory.PLANNING,
        conversation=[
            {"role": "user", "content": "sunday ko trip pe chale?"}
        ],
        expected_topic="plans",
        expected_intent="invitation",
        expected_strategy="accept",
        expected_context_behavior="Excursion proposal",
        expected_use_memory=False,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Day-trip suggestion.",
    ),

    # =========================================================================
    # G. SOCIAL (4 cases)
    # =========================================================================
    BenchmarkCase(
        case_id="CASE_SOCIAL_01",
        category=BenchmarkCategory.SOCIAL,
        conversation=[
            {"role": "user", "content": "Rahul bhi aa raha hai kya sath me?"}
        ],
        expected_topic="social",
        expected_intent="question",
        expected_strategy="answer",
        expected_context_behavior="Mutual acquaintance coordination",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Peer social presence query.",
    ),
    BenchmarkCase(
        case_id="CASE_SOCIAL_02",
        category=BenchmarkCategory.SOCIAL,
        conversation=[
            {"role": "assistant", "content": "kya scene hai"},
            {"role": "user", "content": "party chal rahi hai yahan"}
        ],
        expected_topic="social",
        expected_intent="information",
        expected_strategy="react",
        expected_context_behavior="Social gathering update",
        expected_use_memory=False,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Gathering status report.",
    ),
    BenchmarkCase(
        case_id="CASE_SOCIAL_03",
        category=BenchmarkCategory.SOCIAL,
        conversation=[
            {"role": "user", "content": "bhai treat kab de raha hai?"}
        ],
        expected_topic="social",
        expected_intent="question",
        expected_strategy="answer",
        expected_context_behavior="Playful social demand for treat",
        expected_use_memory=False,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Playful social banter.",
    ),
    BenchmarkCase(
        case_id="CASE_SOCIAL_04",
        category=BenchmarkCategory.SOCIAL,
        conversation=[
            {"role": "assistant", "content": "sab dost mile the"},
            {"role": "user", "content": "bahut maza aaya sabke sath"}
        ],
        expected_topic="social",
        expected_intent="reaction",
        expected_strategy="react",
        expected_context_behavior="Post-reunion sentiment",
        expected_use_memory=False,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Positive peer sentiment.",
    ),

    # =========================================================================
    # H. AMBIGUOUS SHORT MESSAGES (4 cases) - Same phrase under different contexts
    # =========================================================================
    BenchmarkCase(
        case_id="CASE_AMBIG_01",
        category=BenchmarkCategory.AMBIGUOUS_SHORT_MESSAGES,
        conversation=[
            {"role": "user", "content": "Aaja"}
        ],
        expected_topic="casual_chat",
        expected_intent="invitation",
        expected_strategy="ask_clarification",
        expected_context_behavior="Isolated ambiguous imperative without domain cues",
        expected_use_memory=False,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Zero context -> Ask clarification, do not invent gaming.",
    ),
    BenchmarkCase(
        case_id="CASE_AMBIG_02",
        category=BenchmarkCategory.AMBIGUOUS_SHORT_MESSAGES,
        conversation=[
            {"role": "user", "content": "BGMI khelenge?"},
            {"role": "assistant", "content": "haan lobby me hu"},
            {"role": "user", "content": "Aaja"}
        ],
        expected_topic="gaming",
        expected_intent="invitation",
        expected_strategy="accept",
        expected_context_behavior="Gaming context disambiguates 'Aaja' to game lobby",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Contextual gaming interpretation of 'Aaja'.",
    ),
    BenchmarkCase(
        case_id="CASE_AMBIG_03",
        category=BenchmarkCategory.AMBIGUOUS_SHORT_MESSAGES,
        conversation=[
            {"role": "user", "content": "canteen chalte hain bhookh lagi hai"},
            {"role": "assistant", "content": "chalo fir"},
            {"role": "user", "content": "Aaja"}
        ],
        expected_topic="plans",
        expected_intent="invitation",
        expected_strategy="accept",
        expected_context_behavior="Canteen context disambiguates 'Aaja' to campus outing",
        expected_use_memory=False,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Contextual canteen interpretation of 'Aaja' -> NOT gaming.",
    ),
    BenchmarkCase(
        case_id="CASE_AMBIG_04",
        category=BenchmarkCategory.AMBIGUOUS_SHORT_MESSAGES,
        conversation=[
            {"role": "user", "content": "Thik"}
        ],
        expected_topic="casual_chat",
        expected_intent="agreement",
        expected_strategy="acknowledge",
        expected_context_behavior="Short acknowledgment turn",
        expected_use_memory=False,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Isolated affirmation -> Acknowledge.",
    ),

    # =========================================================================
    # I. MULTI-TURN CONTEXT (5 cases) - 1, 3, 5, 7, 11 turns
    # =========================================================================
    BenchmarkCase(
        case_id="CASE_TURN_01",
        category=BenchmarkCategory.MULTI_TURN_CONTEXT,
        conversation=[
            {"role": "user", "content": "kya chal raha?"}
        ],
        expected_topic="casual_chat",
        expected_intent="question",
        expected_strategy="answer",
        expected_context_behavior="1-turn context evaluation",
        expected_use_memory=False,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="1-turn minimal depth.",
    ),
    BenchmarkCase(
        case_id="CASE_TURN_03",
        category=BenchmarkCategory.MULTI_TURN_CONTEXT,
        conversation=[
            {"role": "user", "content": "bhai free hai kya?"},
            {"role": "assistant", "content": "haan bol bhai"},
            {"role": "user", "content": "bgmi aaja"}
        ],
        expected_topic="gaming",
        expected_intent="invitation",
        expected_strategy="accept",
        expected_context_behavior="3-turn context progression into gaming",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="3-turn progression.",
    ),
    BenchmarkCase(
        case_id="CASE_TURN_05",
        category=BenchmarkCategory.MULTI_TURN_CONTEXT,
        conversation=[
            {"role": "user", "content": "kal lab me kyu nahi tha?"},
            {"role": "assistant", "content": "tabiyat theek nahi thi"},
            {"role": "user", "content": "ab kaisa hai?"},
            {"role": "assistant", "content": "ab theek hu"},
            {"role": "user", "content": "prof ne attendance short batai hai"}
        ],
        expected_topic="college",
        expected_intent="information",
        expected_strategy="provide_information",
        expected_context_behavior="5-turn dialogue grounding into college topic",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="5-turn dialogue trajectory.",
    ),
    BenchmarkCase(
        case_id="CASE_TURN_07",
        category=BenchmarkCategory.MULTI_TURN_CONTEXT,
        conversation=[
            {"role": "user", "content": "laptop liya tha na naya?"},
            {"role": "assistant", "content": "haan liya tha"},
            {"role": "user", "content": "kaunsa tha?"},
            {"role": "assistant", "content": "acer nitro"},
            {"role": "user", "content": "performance kaisa hai?"},
            {"role": "assistant", "content": "badhiya chal raha"},
            {"role": "user", "content": "heating problem aa rahi hai kya?"}
        ],
        expected_topic="technology",
        expected_intent="question",
        expected_strategy="answer",
        expected_context_behavior="7-turn technology product assessment",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="7-turn deep context topic maintenance.",
    ),
    BenchmarkCase(
        case_id="CASE_TURN_11",
        category=BenchmarkCategory.MULTI_TURN_CONTEXT,
        conversation=[
            {"role": "user", "content": "kaisa hai"},
            {"role": "assistant", "content": "badhiya"},
            {"role": "user", "content": "college aa raha?"},
            {"role": "assistant", "content": "haan raste me hu"},
            {"role": "user", "content": "assignment laya?"},
            {"role": "assistant", "content": "bag me hai"},
            {"role": "user", "content": "canteen me baithe?"},
            {"role": "assistant", "content": "pehle class attend karte"},
            {"role": "user", "content": "chalo theek"},
            {"role": "assistant", "content": "milte hain class me"},
            {"role": "user", "content": "Dean ne notice lagaya hai notice board pe"}
        ],
        expected_topic="college",
        expected_intent="information",
        expected_strategy="provide_information",
        expected_context_behavior="11-turn full conversational flow keeping college focus",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="11-turn long context stability test.",
    ),

    # =========================================================================
    # J. MEMORY-DEPENDENT (4 cases) - Must leverage memory correctly
    # =========================================================================
    BenchmarkCase(
        case_id="CASE_MEM_DEP_01",
        category=BenchmarkCategory.MEMORY_DEPENDENT,
        conversation=[
            {"role": "user", "content": "gaming session kare aaj?"}
        ],
        expected_topic="gaming",
        expected_intent="invitation",
        expected_strategy="accept",
        expected_context_behavior="Gaming inquiry requiring gaming memory gating",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Must select mem_gaming_pref.",
    ),
    BenchmarkCase(
        case_id="CASE_MEM_DEP_02",
        category=BenchmarkCategory.MEMORY_DEPENDENT,
        conversation=[
            {"role": "user", "content": "mera laptop slow chal raha hai tera kaisa hai?"}
        ],
        expected_topic="technology",
        expected_intent="question",
        expected_strategy="answer",
        expected_context_behavior="Laptop inquiry requiring laptop memory gating",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Must select mem_tech_laptop.",
    ),
    BenchmarkCase(
        case_id="CASE_MEM_DEP_03",
        category=BenchmarkCategory.MEMORY_DEPENDENT,
        conversation=[
            {"role": "user", "content": "Marvel ki series dekhna start karu kya?"}
        ],
        expected_topic="movies",
        expected_intent="question",
        expected_strategy="answer",
        expected_context_behavior="Marvel inquiry requiring movie preference gating",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Must select mem_movie_marvel.",
    ),
    BenchmarkCase(
        case_id="CASE_MEM_DEP_04",
        category=BenchmarkCategory.MEMORY_DEPENDENT,
        conversation=[
            {"role": "user", "content": "Java me project banaye kya?"}
        ],
        expected_topic="technology",
        expected_intent="invitation",
        expected_strategy="accept",
        expected_context_behavior="Java programming inquiry requiring Java preference",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Must select mem_tech_java.",
    ),

    # =========================================================================
    # K. MEMORY-IRRELEVANT (4 cases) - Memories exist but MUST NOT be used
    # =========================================================================
    BenchmarkCase(
        case_id="CASE_MEM_IRR_01",
        category=BenchmarkCategory.MEMORY_IRRELEVANT,
        conversation=[
            {"role": "user", "content": "mausam kaisa hai wahan?"}
        ],
        expected_topic="casual_chat",
        expected_intent="question",
        expected_strategy="answer",
        expected_context_behavior="Weather question has zero relevance to tech/gaming memories",
        expected_use_memory=False,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Must NOT select any gaming or tech memories.",
    ),
    BenchmarkCase(
        case_id="CASE_MEM_IRR_02",
        category=BenchmarkCategory.MEMORY_IRRELEVANT,
        conversation=[
            {"role": "user", "content": "chai peene chalte hain"}
        ],
        expected_topic="casual_chat",
        expected_intent="invitation",
        expected_strategy="accept",
        expected_context_behavior="Tea break outing without memory relevance",
        expected_use_memory=False,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Must NOT leak laptop or college memories.",
    ),
    BenchmarkCase(
        case_id="CASE_MEM_IRR_03",
        category=BenchmarkCategory.MEMORY_IRRELEVANT,
        conversation=[
            {"role": "user", "content": "koi acchi jagah bata ghumne ki"}
        ],
        expected_topic="casual_chat",
        expected_intent="question",
        expected_strategy="answer",
        expected_context_behavior="Travel recommendation query",
        expected_use_memory=False,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Must NOT inject Java or Marvel memories.",
    ),
    BenchmarkCase(
        case_id="CASE_MEM_IRR_04",
        category=BenchmarkCategory.MEMORY_IRRELEVANT,
        conversation=[
            {"role": "user", "content": "yaar bohot neend aa rahi hai"}
        ],
        expected_topic="casual_chat",
        expected_intent="information",
        expected_strategy="react",
        expected_context_behavior="Tiredness comment",
        expected_use_memory=False,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Zero memory utilization allowed.",
    ),

    # =========================================================================
    # L. CLOSING (4 cases) - Must wrap up cleanly and suppress memories
    # =========================================================================
    BenchmarkCase(
        case_id="CASE_CLOSE_01",
        category=BenchmarkCategory.CLOSING,
        conversation=[
            {"role": "user", "content": "Chal baad me baat karta hu"}
        ],
        expected_topic="casual_chat",
        expected_intent="planning",
        expected_strategy="close_conversation",
        expected_context_behavior="Dialogue wrap-up sign-off",
        expected_use_memory=False,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Must select close_conversation strategy.",
    ),
    BenchmarkCase(
        case_id="CASE_CLOSE_02",
        category=BenchmarkCategory.CLOSING,
        conversation=[
            {"role": "assistant", "content": "sahi hai bhai"},
            {"role": "user", "content": "bye bhai goodnight"}
        ],
        expected_topic="casual_chat",
        expected_intent="agreement",
        expected_strategy="close_conversation",
        expected_context_behavior="Night sign-off",
        expected_use_memory=False,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Bedtime sign-off.",
    ),
    BenchmarkCase(
        case_id="CASE_CLOSE_03",
        category=BenchmarkCategory.CLOSING,
        conversation=[
            {"role": "user", "content": "see you tomorrow bhai chalta hu"}
        ],
        expected_topic="casual_chat",
        expected_intent="planning",
        expected_strategy="close_conversation",
        expected_context_behavior="Departure departure note",
        expected_use_memory=False,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Departure wrap-up.",
    ),
    BenchmarkCase(
        case_id="CASE_CLOSE_04",
        category=BenchmarkCategory.CLOSING,
        conversation=[
            {"role": "assistant", "content": "theek hai"},
            {"role": "user", "content": "tata bye"}
        ],
        expected_topic="casual_chat",
        expected_intent="agreement",
        expected_strategy="close_conversation",
        expected_context_behavior="Short closing token",
        expected_use_memory=False,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Short departure affirmation.",
    ),

    # =========================================================================
    # M. DISAGREEMENT / REJECTION (4 cases)
    # =========================================================================
    BenchmarkCase(
        case_id="CASE_DISAGREE_01",
        category=BenchmarkCategory.DISAGREEMENT_REJECTION,
        conversation=[
            {"role": "user", "content": "nhi bhai aisa nahi hai"}
        ],
        expected_topic="casual_chat",
        expected_intent="disagreement",
        expected_strategy="react",
        expected_context_behavior="Negative dispute of statement",
        expected_use_memory=False,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Standard disagreement.",
    ),
    BenchmarkCase(
        case_id="CASE_DISAGREE_02",
        category=BenchmarkCategory.DISAGREEMENT_REJECTION,
        conversation=[
            {"role": "assistant", "content": "game aaja abhi"},
            {"role": "user", "content": "nhi bhai abhi nahi aa sakta"}
        ],
        expected_topic="gaming",
        expected_intent="invitation",
        expected_strategy="decline",
        expected_context_behavior="Direct rejection of gaming invitation",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Decline gaming invitation.",
    ),
    BenchmarkCase(
        case_id="CASE_DISAGREE_03",
        category=BenchmarkCategory.DISAGREEMENT_REJECTION,
        conversation=[
            {"role": "user", "content": "galat logic hai bhai code me"}
        ],
        expected_topic="technology",
        expected_intent="disagreement",
        expected_strategy="answer",
        expected_context_behavior="Technical code critique",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Technical disagreement warrants answer/explanation.",
    ),
    BenchmarkCase(
        case_id="CASE_DISAGREE_04",
        category=BenchmarkCategory.DISAGREEMENT_REJECTION,
        conversation=[
            {"role": "user", "content": "nahi yaar bore lag raha hai"}
        ],
        expected_topic="casual_chat",
        expected_intent="disagreement",
        expected_strategy="react",
        expected_context_behavior="Disinterest reaction",
        expected_use_memory=False,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Discontentment expression.",
    ),

    # =========================================================================
    # N. QUESTIONS (4 cases)
    # =========================================================================
    BenchmarkCase(
        case_id="CASE_Q_01",
        category=BenchmarkCategory.QUESTIONS,
        conversation=[
            {"role": "user", "content": "Exam kab se start ho rahe hain?"}
        ],
        expected_topic="college",
        expected_intent="question",
        expected_strategy="answer",
        expected_context_behavior="Academic inquiry",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Clear academic question.",
    ),
    BenchmarkCase(
        case_id="CASE_Q_02",
        category=BenchmarkCategory.QUESTIONS,
        conversation=[
            {"role": "user", "content": "Python me list comprehension kaise likhte hain?"}
        ],
        expected_topic="technology",
        expected_intent="question",
        expected_strategy="answer",
        expected_context_behavior="Technical programming question",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Programming language question.",
    ),
    BenchmarkCase(
        case_id="CASE_Q_03",
        category=BenchmarkCategory.QUESTIONS,
        conversation=[
            {"role": "user", "content": "konsa character choose karu match me?"}
        ],
        expected_topic="gaming",
        expected_intent="question",
        expected_strategy="answer",
        expected_context_behavior="Gaming character selection inquiry",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Game strategy question.",
    ),
    BenchmarkCase(
        case_id="CASE_Q_04",
        category=BenchmarkCategory.QUESTIONS,
        conversation=[
            {"role": "user", "content": "movie ka climax kaisa tha?"}
        ],
        expected_topic="movies",
        expected_intent="question",
        expected_strategy="answer",
        expected_context_behavior="Plot inquiry",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Entertainment narrative question.",
    ),

    # =========================================================================
    # O. INVITATIONS (4 cases)
    # =========================================================================
    BenchmarkCase(
        case_id="CASE_INV_01",
        category=BenchmarkCategory.INVITATIONS,
        conversation=[
            {"role": "user", "content": "Khelega?"}
        ],
        expected_topic="gaming",
        expected_intent="invitation",
        expected_strategy="accept",
        expected_context_behavior="Direct gaming invitation",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Direct gaming proposal.",
    ),
    BenchmarkCase(
        case_id="CASE_INV_02",
        category=BenchmarkCategory.INVITATIONS,
        conversation=[
            {"role": "user", "content": "canteen chale?"}
        ],
        expected_topic="plans",
        expected_intent="invitation",
        expected_strategy="accept",
        expected_context_behavior="Canteen outing invitation",
        expected_use_memory=False,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Casual campus invitation.",
    ),
    BenchmarkCase(
        case_id="CASE_INV_03",
        category=BenchmarkCategory.INVITATIONS,
        conversation=[
            {"role": "user", "content": "kal cinema chalega?"}
        ],
        expected_topic="movies",
        expected_intent="invitation",
        expected_strategy="accept",
        expected_context_behavior="Cinema invitation",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Film attendance invitation.",
    ),
    BenchmarkCase(
        case_id="CASE_INV_04",
        category=BenchmarkCategory.INVITATIONS,
        conversation=[
            {"role": "user", "content": "bhai aaja squad ready hai"}
        ],
        expected_topic="gaming",
        expected_intent="invitation",
        expected_strategy="accept",
        expected_context_behavior="Gaming team invitation",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Team gaming call.",
    ),

    # =========================================================================
    # P. UNSEEN / GENERALIZATION (4 cases) - Novel phrasing & code-switching
    # =========================================================================
    BenchmarkCase(
        case_id="CASE_UNSEEN_01",
        category=BenchmarkCategory.UNSEEN_GENERALIZATION,
        conversation=[
            {"role": "user", "content": "bro CPU temperature is hitting 95C during rendering"}
        ],
        expected_topic="technology",
        expected_intent="information",
        expected_strategy="suggest",
        expected_context_behavior="English hardware diagnostics problem",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Thermal diagnostic statement.",
    ),
    BenchmarkCase(
        case_id="CASE_UNSEEN_02",
        category=BenchmarkCategory.UNSEEN_GENERALIZATION,
        conversation=[
            {"role": "user", "content": "HOD called for emergency department meet today"}
        ],
        expected_topic="college",
        expected_intent="information",
        expected_strategy="provide_information",
        expected_context_behavior="English academic administrative notice",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Departmental administrative alert.",
    ),
    BenchmarkCase(
        case_id="CASE_UNSEEN_03",
        category=BenchmarkCategory.UNSEEN_GENERALIZATION,
        conversation=[
            {"role": "user", "content": "marvel cinematic universe is going downhill lately"}
        ],
        expected_topic="movies",
        expected_intent="information",
        expected_strategy="react",
        expected_context_behavior="Entertainment franchise editorial remark",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Critical pop-culture sentiment.",
    ),
    BenchmarkCase(
        case_id="CASE_UNSEEN_04",
        category=BenchmarkCategory.UNSEEN_GENERALIZATION,
        conversation=[
            {"role": "user", "content": "ping me whenever you are free to deploy"}
        ],
        expected_topic="technology",
        expected_intent="planning",
        expected_strategy="suggest",
        expected_context_behavior="DevOps software deployment coordination",
        expected_use_memory=True,
        available_memories=STANDARD_MEMORY_BANK,
        evaluation_notes="Software deployment timing coordination.",
    ),
]
