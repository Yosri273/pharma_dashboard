from typing import List


def generate_recommendations(insights: list[str]) -> list[str]:
    """Generate actionable recommendations from a list of text insights.

    This function is pure and has no side effects. It accepts a list of
    factual insight strings and returns a list of pre-written, actionable
    recommendation strings. Rules are matched using simple keyword checks
    and pattern membership (case-insensitive). If no rule matches any
    insight, a single positive default recommendation is returned.

    Args:
        insights: A list of factual observation strings about the business.

    Returns:
        A list of actionable recommendation strings.
    """

    # Normalize input to ensure case-insensitive matching
    if not insights:
        return [
            "**Recommendation:** All key metrics appear to be within their target ranges. Continue monitoring and consider exploring new growth opportunities."
        ]

    recommendations: List[str] = []

    for raw in insights:
        if not raw or not isinstance(raw, str):
            continue

        s = raw.lower()

        # Rule: low conversion rate
        if "conversion rate is below" in s or "low conversion rate" in s or "conversion rate below" in s:
            recommendations.append(
                "**Recommendation:** A/B test the checkout page design. Experiment with button colors and reducing the number of form fields to decrease friction."
            )
            continue

        # Rule: best-selling product
        if "is the best-selling product" in s or "best selling product" in s or "top-selling product" in s:
            recommendations.append(
                "**Recommendation:** Boost the top-selling product by launching a targeted ad campaign on your highest-performing marketing channel. Consider creating a product bundle around it."
            )
            continue

        # Rule: cart abandonment
        if "cart abandonment rate" in s or "abandonment rate" in s and "cart" in s:
            recommendations.append(
                "**Recommendation:** Review the mobile checkout flow for potential friction points. A high abandonment rate often indicates a complex or slow process."
            )
            continue

        # Rule: declining retention or churn
        if "churn" in s or "retention is down" in s or "customer retention" in s and ("down" in s or "decline" in s or "decreasing" in s):
            recommendations.append(
                "**Recommendation:** Implement a personalized retention program (email + in-app messaging) for at-risk cohorts. Offer targeted incentives and survey users to identify friction."
            )
            continue

        # Rule: rising acquisition cost or high CAC
        if "cost per acquisition" in s or "cpa" in s or "acquisition cost" in s or "high cac" in s:
            recommendations.append(
                "**Recommendation:** Re-evaluate the marketing channel mix and pause underperforming campaigns. Shift budget toward channels with lower CAC and higher LTV. Test lookalike audiences to improve efficiency."
            )
            continue

        # Rule: inventory or supply issues
        if "out of stock" in s or "inventory" in s and ("low" in s or "shortage" in s or "stockout" in s):
            recommendations.append(
                "**Recommendation:** Review supplier lead times and safety stock policies. Prioritize replenishment for high-velocity SKUs and consider temporary substitutes or back-in-stock alerts."
            )
            continue

        # Rule: slowdown in site traffic
        if "traffic" in s and ("down" in s or "decreased" in s or "declining" in s):
            recommendations.append(
                "**Recommendation:** Audit recent marketing campaigns and SEO changes. Re-activate high-performing channels and run short paid tests to recover volume."
            )
            continue

        # Rule: slow page load or performance issues
        if "page load" in s or "slow" in s and ("page" in s or "site" in s or "load time" in s):
            recommendations.append(
                "**Recommendation:** Work with engineering to profile front-end assets and optimize images, lazy-load non-critical resources, and enable caching/CDN where possible."
            )
            continue

        # Rule: promo or discount performing well/poorly
        if "promotion" in s or "discount" in s or "coupon" in s:
            if "performing well" in s or "lift" in s or "increase" in s:
                recommendations.append(
                    "**Recommendation:** Scale successful promotions to similar customer segments and capture learnings for future campaigns. Ensure margins remain acceptable."
                )
            else:
                recommendations.append(
                    "**Recommendation:** Reassess promotion targeting and messaging. Use controlled experiments to measure incremental lift and prevent margin erosion."
                )
            continue

        # Generic heuristic: suggest A/B testing where KPI language appears
        if any(k in s for k in ("test", "experiment", "lift", "impact")) or any(k in s for k in ("increase", "decrease", "improve", "drop")):
            recommendations.append(
                "**Recommendation:** Design a focused experiment (A/B test) that isolates the suspected cause. Define primary metrics, sample size, and duration before running."
            )
            continue

        # If we reach here, no rule matched this insight specifically; add a gentle, actionable suggestion
        recommendations.append(
            "**Recommendation:** Review the related metric in more detail and run a small investigation to identify root causes and quick wins."
        )

    # If we matched nothing at all, provide a positive default
    if not recommendations:
        return [
            "**Recommendation:** All key metrics appear to be within their target ranges. Continue monitoring and consider exploring new growth opportunities."
        ]

    return recommendations
