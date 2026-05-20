process_sentences_e = {"activated": [
                            "All {species} in the team can see who takes over tasks and how they perform.",
                            "Your decision and your performance on any task you take over will be visible to everyone in the team.",
                            "Your response to this message is visible to all {species} in the team, who will also be able to see how you handle the task.",
                            "All {species} in the team are aware of who responds to requests and how well they handle the additional work.",
                            "The entire team will be able to see who volunteers to help and how effectively they complete the work they take on.",
                        ],
                       "blocked": [
                            "Your decision will not be visible to the other {species} in the team. ",
                            "Your response and actions will remain private and will not be seen by the rest of the team. ",
                            "Your decision will be kept private and you will not be evaluated or judged by others. ",
                            "Neither your decision nor your performance will be visible to the rest of the team. ",
                            "Your choice will remain confidential and will not be shared with or judged by the other {species}. "
                       ]
}

process_sentences_r = {"activated":
                       [
                            "You are not the only one who could take over. Other {species} in the team may also decide to take over {requester_name}'s subtask.",
                            "Other {species} in the team are as well capable of taking over {requester_name}'s subtask and may do so.",
                            "The responsibility to take over {requester_name}'s work is shared among all {species} in the team, you are not the only one.",
                            "Taking over {requester_name}'s subtask is not solely your responsibility, other {species} in the team share this responsibility equally.",
                            "There are other {species} in the team who are available and could take over {requester_name}'s remaining work."
                        ],
                       "blocked":
                       [
                           "You are the only {specy} with the expertise needed to complete {requester_name}'s subtask. ",
                           "Only you have the knowledge required to take over {requester_name}'s work: no other {specy} in the team is in a position to help. ",
                           "No other {specy} in the team has the necessary expertise to complete {requester_name}'s subtask — the responsibility rests solely with you. ",
                           "You are uniquely positioned to take over {requester_name}'s work, as no other {specy} in the team has the required expertise. ",
                           "The rest of the team lacks the expertise to complete {requester_name}'s subtask, making you the only one capable of helping. "
                       ]
                      }


process_sentences_p = {"activated":
                       [
                            "You can observe that none of the other {species} have responded yet. It is unclear whether they consider the situation needs intervention.",
                            "You notice that the other {species} in the team have not yet reacted to {requester_name}'s request. It is unclear whether they think the situation requires action.",
                            "None of the other {species} have stepped in yet. Whether this means they do not consider the situation serious, or simply have not decided yet, is unclear.",
                            "The other {species} in the team have not yet signaled whether they intend to help {requester_name}. It is uncertain what the appropriate response is in this situation.",
                            "None of the other {species} have reacted to {requester_name}'s message, making it difficult to gauge whether taking over is expected or necessary."
                       ],
                       "blocked":
                       [
                           "The situation is widely recognized as critical and requiring intervention.",
                            "It is clear to everyone that {requester_name}'s situation requires someone to step in.",
                            "The urgency of the situation leaves no ambiguity about whether intervention is needed."
                            "The situation is unambiguously critical and widely understood to require action.",
                             "There is no uncertainty about the severity of the situation: it is clear that action is needed."
                       ]
}