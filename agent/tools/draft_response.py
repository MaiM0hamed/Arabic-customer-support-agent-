"""Response drafting tool: generates an Arabic reply via the LLM."""

import logging
from pathlib import Path
from typing import Any

from llm.client import OpenRouterClient, OpenRouterError

logger = logging.getLogger(__name__)

_FALLBACK_RESPONSES: dict[str, str] = {
    "order_status": "شكرًا لتواصلك معنا. تم استلام استفسارك بخصوص حالة طلبك وسيقوم فريقنا بالتحقق من آخر تحديث وإعلامك في أقرب وقت.",
    "shipping_delay": "نعتذر عن التأخير في توصيل طلبك. تم تسجيل طلبك وسيقوم فريق اللوجستيات بالمتابعة مع شركة الشحن وإفادتك بالمستجدات.",
    "refund_request": "تم استلام طلب الاسترداد الخاص بك، وسيقوم فريق الإرجاع والاسترداد بمراجعته ومعالجته خلال المدة المحددة في سياسة الاسترداد.",
    "return_request": "تم استلام طلب الإرجاع الخاص بك، يرجى اتباع الخطوات الموضحة في صفحة طلباتي لإتمام عملية الإرجاع.",
    "exchange_request": "تم استلام طلب الاستبدال الخاص بك، وسنقوم بترتيب استلام المنتج الحالي وتوصيل البديل في أقرب وقت.",
    "damaged_product": "نأسف لاستلامك المنتج بهذه الحالة. تم تسجيل بلاغك وسيقوم فريقنا بالتواصل معك لترتيب الاستبدال أو الاسترداد.",
    "payment_issue": "نعتذر عن أي إزعاج بخصوص عملية الدفع. تم تحويل المشكلة إلى فريق المدفوعات للتحقق من الخصم ومعالجته.",
    "app_bug": "نأسف للمشكلة التقنية التي واجهتها. تم تحويل بلاغك إلى فريق الدعم التقني للعمل على حلها في أقرب وقت.",
    "account_help": "تم استلام طلبك المتعلق بحسابك، وسيقوم فريق دعم الحسابات بمساعدتك في استعادة الوصول إلى حسابك.",
    "general_inquiry": "شكرًا لتواصلك معنا. سنقوم بالرد على استفسارك في أقرب وقت ممكن.",
    "complaint": "نأسف لتجربتك ونقدر تواصلك معنا. تم تسجيل شكواك وسيتم متابعتها من قبل فريق رعاية العملاء.",
    "contact_support_request": "تم تحويل طلبك إلى أحد ممثلي خدمة العملاء وسيتم التواصل معك في أقرب وقت ممكن.",
}

_DEFAULT_FALLBACK = "شكرًا لتواصلك معنا، تم استلام رسالتك وسيتم الرد عليك في أقرب وقت ممكن."

# Per-dialect instructions injected into the response prompt so the reply's
# register matches the customer's dialect instead of defaulting to MSA.
_DIALECT_GUIDANCE: dict[str, str] = {
    "msa": "اكتب الرد بالعربية الفصحى المبسطة (الفصحى المعاصرة) المستخدمة عادة في رسائل خدمة العملاء الرسمية.",
    "egyptian": (
        "اكتب الرد باللهجة المصرية العامية المهذبة، باستخدام كلمات وتعبيرات مصرية شائعة "
        "(مثل: حضرتك، تمام، هنقوم بكذا، عشان، علشان، هنتواصل معك) مع الحفاظ على الاحترافية. "
        "لا تستخدم الفصحى الرسمية."
    ),
    "gulf": (
        "اكتب الرد باللهجة الخليجية المهذبة، باستخدام كلمات وتعبيرات خليجية شائعة "
        "(مثل: نسعد بخدمتك، إن شاء الله، أبشر، تراني بتابع معك، حياك الله) مع الحفاظ على الاحترافية. "
        "لا تستخدم الفصحى الرسمية."
    ),
    "levantine": (
        "اكتب الرد باللهجة الشامية (الشام) المهذبة، باستخدام كلمات وتعبيرات شامية شائعة "
        "(مثل: تمام، هلق، رح نتابع معك، منيح، إن شاء الله) مع الحفاظ على الاحترافية. "
        "لا تستخدم الفصحى الرسمية."
    ),
    "maghrebi": (
        "اكتب الرد باللهجة المغربية المهذبة، باستخدام كلمات وتعبيرات مغربية شائعة "
        "(مثل: غادي نتابعو معاك، دابا، بغيتي، إن شاء الله) مع الحفاظ على الاحترافية. "
        "لا تستخدم الفصحى الرسمية."
    ),
}
_DEFAULT_DIALECT_GUIDANCE = _DIALECT_GUIDANCE["msa"]

_PROMPT_TEMPLATE = Path("llm/prompts/response_prompt.txt").read_text(encoding="utf-8")
_SYSTEM_PROMPT = Path("llm/prompts/system_prompt.txt").read_text(encoding="utf-8")


def draft_response(
    message: str,
    intent: str,
    dialect: str,
    context: list[dict[str, Any]] | None = None,
    llm_client: OpenRouterClient | None = None,
) -> str:
    """Draft an Arabic customer-support reply.

    Attempts to generate a reply via the LLM using the matched dialect and
    knowledge-base context, falling back to a static template if the LLM
    call fails.

    Args:
        message: Sanitized customer message text.
        intent: Classified intent identifier.
        dialect: Detected dialect identifier.
        context: Optional list of knowledge-base documents to ground the
            reply (each with a `content_ar` field).
        llm_client: Optional `OpenRouterClient` instance. If `None`, the
            static fallback template is used.

    Returns:
        str: A non-empty Arabic reply string.
    """
    fallback = _FALLBACK_RESPONSES.get(intent, _DEFAULT_FALLBACK)

    if llm_client is None:
        return fallback

    try:
        context_str = "\n".join(f"- {doc.get('content_ar', '')}" for doc in (context or []))

        prompt = _PROMPT_TEMPLATE.format(
            dialect=dialect,
            dialect_instruction=_DIALECT_GUIDANCE.get(dialect, _DEFAULT_DIALECT_GUIDANCE),
            intent=intent,
            context=context_str or "لا يوجد سياق إضافي.",
            message=message,
        )

        response = llm_client.chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.4,
        )
        result = llm_client.extract_json(response)
        reply = str(result.get("response_ar") or "").strip()
        return reply or fallback
    except (OpenRouterError, OSError, KeyError, AttributeError) as exc:
        logger.warning("LLM response drafting failed: %s", exc)
        return fallback
