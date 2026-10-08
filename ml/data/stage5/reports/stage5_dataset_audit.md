# Stage 5 — Dataset Audit & Integrity Report

**Generated:** 2026-10-08 15:27:53  
**Total Stage 5 Examples:** 5,938  
**Split Counts:** Train: 3,760 | Val: 868 | Test: 1,310  
**Unique Conversations:** Train: 354 | Val: 77 | Test: 77 (Total: 508)  

## 1. Integrity Verification Summary

| Integrity Check | Expected | Actual | Status |
|:---|:---:|:---:|:---:|
| Missing Target Responses | 0 | 0 | **PASS** |
| Target Response Modifications | 0 | 0 | **PASS** |
| Train ∩ Val Conversation Overlap | 0 | 0 | **PASS** |
| Train ∩ Test Conversation Overlap | 0 | 0 | **PASS** |
| Val ∩ Test Conversation Overlap | 0 | 0 | **PASS** |
| Duplicate Dialogue Examples | 0 | 0 | **PASS** |
| Field Completeness (Behavioral, Topic, Context) | 100% | 100% | **PASS** |

## 2. Behavioral Distributions

### Language Distribution
| Language | Count | Percentage |
|:---|:---:|:---:|
| `hinglish` | 5,569 | 93.79% |
| `english` | 268 | 4.51% |
| `mixed` | 60 | 1.01% |
| `unknown` | 41 | 0.69% |

### Tone Distribution
| Tone | Count | Percentage |
|:---|:---:|:---:|
| `neutral` | 3,501 | 58.96% |
| `casual` | 1,626 | 27.38% |
| `teasing` | 260 | 4.38% |
| `serious` | 236 | 3.97% |
| `uncertain` | 200 | 3.37% |
| `humorous` | 82 | 1.38% |
| `supportive` | 33 | 0.56% |

### Response Type Distribution
| Response Type | Count | Percentage |
|:---|:---:|:---:|
| `statement` | 3,475 | 58.52% |
| `answer` | 1,345 | 22.65% |
| `question` | 346 | 5.83% |
| `acknowledgement` | 345 | 5.81% |
| `suggestion` | 192 | 3.23% |
| `invitation` | 118 | 1.99% |
| `refusal` | 104 | 1.75% |
| `reaction` | 13 | 0.22% |

### Stylistic Markers
| Marker | Count | Percentage |
|:---|:---:|:---:|
| Has Slang / Colloquialisms | 1,003 | 16.89% |
| Has Emoji | 766 | 12.90% |
| Is Question | 346 | 5.83% |

## 3. Topic Distributions

| Topic Category | Count | Percentage |
|:---|:---:|:---:|
| `casual_chat` | 2,700 | 45.47% |
| `plans` | 900 | 15.16% |
| `gaming` | 694 | 11.69% |
| `technology` | 614 | 10.34% |
| `social` | 415 | 6.99% |
| `college` | 229 | 3.86% |
| `other` | 199 | 3.35% |
| `sports` | 143 | 2.41% |
| `movies` | 44 | 0.74% |

## 4. Context & Structural Dynamics

### Context Depth Distribution
| Context Messages | Count | Percentage |
|:---:|:---:|:---:|
| 1 | 2,300 | 38.73% |
| 3 | 1,582 | 26.64% |
| 7 | 1,188 | 20.01% |
| 11 | 868 | 14.62% |

### Inferred Conversation State
| State | Count | Percentage |
|:---|:---:|:---:|
| `ongoing` | 2,913 | 49.06% |
| `banter` | 920 | 15.49% |
| `inquiry` | 824 | 13.88% |
| `opening` | 773 | 13.02% |
| `agreement` | 426 | 7.17% |
| `closing` | 82 | 1.38% |

## 5. Emoji & Length Statistics

- **Total Emojis in Dataset:** 816
- **Emoji Occurrence Rate:** 12.90% of messages contain emojis
- **Top Emojis:** 🙂 (110), 👾 (88), 👍🏻 (62), 🙃 (54), 🗿 (50), 👽 (43), 😂 (37), 🥲 (34), 🫂 (33), 💀 (31)

### Response Length Metrics
- **Average Characters / Message:** 40.01 (Median: 28)
- **Average Words / Message:** 8.21 (Median: 6)

| Length Category | Count | Percentage |
|:---|:---:|:---:|
| `medium` | 2,575 | 43.36% |
| `short` | 1,404 | 23.64% |
| `long` | 1,120 | 18.86% |
| `very_short` | 839 | 14.13% |

## 6. Ambiguous Prompts Benchmark Summary

- **Total Ambiguous Prompts Identified:** 67
| Prompt | Distinct Target Responses | Total Occurrences |
|:---|:---:|:---:|
| `Aaja` | 10 | 52 |
| `Aaja Bhai` | 4 | 6 |
| `Aaja Game M` | 4 | 8 |
| `Aaja Re` | 3 | 6 |
| `Game Aaja` | 2 | 3 |
| `Aaja Be` | 2 | 4 |
| `Aaja Bhadwe` | 2 | 6 |
| `Lawde Aaja` | 2 | 5 |
| `Khelega` | 10 | 36 |
| `Khelega Kya` | 2 | 2 |
