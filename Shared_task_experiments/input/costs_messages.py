cost_blocks = [
    # Version 1 - original symmetric version
    (
        "If you take over {requester_name}'s remaining work:\n"
        "- You will need to pause your own subtask, increasing the risk "
        "that your work is unfinished by the deadline.\n"
        "- You may not have sufficient information to complete "
        "{requester_name}'s subtask successfully.\n\n"
        "If you do not take over {requester_name}'s remaining work:\n"
        "- {requester_name}'s subtask will remain incomplete, "
        "increasing the risk that their work is unfinished by the deadline.\n"
        "- No other may have sufficient information to complete "
        "{requester_name}'s subtask successfully.\n"
    ),

    # Version 2 - progress framing
    (
        "If you take over {requester_name}'s remaining work:\n"
        "- Your own subtask will be delayed, putting your progress at risk.\n"
        "- You may lack the necessary background to successfully complete "
        " {requester_name}'s work.\n\n"
        "If you do not take over {requester_name}'s remaining work:\n"
        "- {requester_name}'s subtask may not be completed in time.\n"
        "- It is uncertain whether anyone else has the background needed "
        "to successfully complete {requester_name}'s work.\n"
    ),

    # Version 3 - deadline framing
    (
        "Taking over {requester_name}'s work means:\n"
        "- Your own deadline becomes harder to meet as you pause your current subtask.\n"
        "- There is no guarantee that you have enough context to finish "
        "{requester_name}'s subtask successfully.\n\n"
        "Not taking over {requester_name}'s work means:\n"
        "- The deadline for {requester_name}'s subtask may be missed.\n"
        "- There is no guarantee that anyone else has enough context to finish "
        " {requester_name}'s subtask successfully.\n"
    ),

    # Version 4 - risk framing
    (
        "Choosing to take over {requester_name}'s remaining work carries the following risks:\n"
        "- Your own subtask may remain unfinished as a result of the interruption.\n"
        "- You may not have the information required to complete "
        "{requester_name}'s subtask to the necessary standard.\n\n"
        "Choosing not to take over {requester_name}'s remaining work carries the following risks:\n"
        "- {requester_name}'s subtask may remain unfinished by the deadline.\n"
        "- No other {specy} may have the information required to complete "
        "{requester_name}'s subtask to the necessary standard.\n"
    ),

    # Version 5 - trade-off framing
    (
        "There is a trade-off to consider:\n"
        "- If you step in, you will need to set aside your own subtask, "
        "which may affect your ability to meet your own deadline. "
        "It is also uncertain whether you have sufficient context to complete "
        "{requester_name}'s work.\n"
        "- If you do not step in, {requester_name}'s subtask risks "
        "being unfinished by the deadline. "
        "It is equally uncertain whether any other {specy} has sufficient context "
        "to complete {requester_name}'s work.\n"
    ),
]

requester_messages_group = [
    # Version 1
    "Can someone take over the rest of my subtask? "
    "I am stuck and do not know if I will finish in time. "
    "I will transfer the remaining files.",
    # Version 2
    "I am not making enough progress on my subtask and am worried I will not meet the deadline. "
    "Is anyone able to take over from where I left off? "
    "I can hand over all the relevant files.",
    # Version 3
    "I am struggling with my subtask and am not sure I can finish it in time. "
    "Would anyone be able to step in and take over? "
    "I can pass on the remaining files right away.",
    # Version 4
    "I could really use some help with my subtask. I am falling behind and may not finish in time. "
    "If anyone is able to take over, I will send the remaining files straight away.",
    # Version 5
    "I am behind on my subtask due to unforeseen problems and need someone to take over. "
    "I will transfer the files to whoever can help.",
]

requester_messages_solo = [
    # Version 1
    "Could you take over the rest of my subtask? "
    "I am stuck and do not know if I will finish in time. "
    "I will transfer the remaining files.",
    # Version 2
    "I am not making enough progress on my subtask and am worried I will not meet the deadline. "
    "Are you able to take over from where I left off? "
    "I can hand over all the relevant files.",
    # Version 3
    "I am struggling with my subtask and am not sure I can finish it in time. "
    "Would you be able to step in and take over? "
    "I can pass on the remaining files right away.",
    # Version 4
    "I could really use your help with my subtask. I am falling behind and may not finish in time. "
    "If you are able to take over, I will send the remaining files straight away.",
    # Version 5
    "I am behind on my subtask due to unforeseen problems and need you to take over. "
    "I will transfer the files to you right away.",
]