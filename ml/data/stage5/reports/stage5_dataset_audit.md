# Stage 5 — Dataset Audit & Integrity Report

**Generated:** 2026-10-08 14:44:56  
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
| `casual` | 4,595 | 77.38% |
| `neutral` | 601 | 10.12% |
| `teasing` | 222 | 3.74% |
| `serious` | 201 | 3.38% |
| `uncertain` | 193 | 3.25% |
| `humorous` | 116 | 1.95% |
| `supportive` | 10 | 0.17% |

### Response Type Distribution
| Response Type | Count | Percentage |
|:---|:---:|:---:|
| `statement` | 2,898 | 48.80% |
| `answer` | 2,084 | 35.10% |
| `question` | 346 | 5.83% |
| `acknowledgement` | 321 | 5.41% |
| `suggestion` | 104 | 1.75% |
| `refusal` | 94 | 1.58% |
| `invitation` | 78 | 1.31% |
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
| `casual_chat` | 2,424 | 40.82% |
| `plans` | 1,286 | 21.66% |
| `technology` | 634 | 10.68% |
| `gaming` | 521 | 8.77% |
| `social` | 432 | 7.28% |
| `college` | 221 | 3.72% |
| `other` | 220 | 3.70% |
| `sports` | 137 | 2.31% |
| `movies` | 63 | 1.06% |

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

- **Total Emojis in Dataset:** 934
- **Emoji Occurrence Rate:** 12.90% of messages contain emojis
- **Top Emojis:** 🏻 (118), 🙂 (110), 👾 (88), 👍 (71), 🙃 (54), 🗿 (50), 👽 (43), 🕺 (40), 😂 (37), 🥲 (34)

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
