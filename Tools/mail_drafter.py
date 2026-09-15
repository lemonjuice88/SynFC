"""
mail_drafter.py — formal correspondence DRAFTING (not sending)
====================================================================
Generates formal letter/email text (e.g. to UEFA, FIFA, or another
governing body) for the Legal department. Deliberately DRAFT-ONLY --
this does NOT send anything, has no SMTP/email-sending code at all.

Why: SynFC doesn't represent a real club, and actually sending mail
to real institutional addresses (even accidentally, even in testing)
isn't something an autonomous agent tool should be able to trigger on
its own. A human should read the draft and decide whether/where to
actually send it, by hand, outside of this system. See
SYNFC_VISION.md's "action-approval UX" open question -- this is a
first, deliberately conservative step in that direction.
"""

import core


def draft_formal_letter(topic: str, recipient: str, club_name: str) -> str:
    """Uses the LLM to draft a formal letter, in the register expected
    for correspondence with a football governing body. Returns TEXT
    ONLY -- no sending, no attachments, no side effects.
    """
    system_prompt = """\
You draft formal, professional correspondence on behalf of a football
club's legal department, addressed to a football governing body (e.g.
FIFA, UEFA, a national federation). Register: formal, precise, no
casual language, no emojis. Standard business-letter structure:
salutation, clear statement of purpose in the first paragraph, body,
formal closing. Do not invent specific rule article numbers or facts
you haven't been given -- if the topic requires citing a specific
regulation, leave a clear placeholder like "[cite relevant article]"
rather than making one up.
"""
    user_prompt = (
        f"Draft a formal letter from {club_name} to {recipient} regarding "
        f"the following matter:\n\n{topic}"
    )
    return core.call_llm(system_prompt, user_prompt)


if __name__ == "__main__":
    # Quick manual test:
    #   python mail_drafter.py
    draft = draft_formal_letter(
        topic="Requesting clarification on training compensation calculation for a recent transfer.",
        recipient="FIFA Players' Status Department",
        club_name="Example FC",
    )
    print(draft)