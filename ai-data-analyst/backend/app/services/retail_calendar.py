"""Retail reporting intervals. Every boundary is midnight UTC and end-exclusive."""
import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta

PATTERNS = ("4-4-5", "4-5-4", "5-4-4")
SETTING_KEYS = ("retail_pattern", "retail_preset", "retail_month", "retail_weekday",
                "retail_rule", "retail_label", "retail_extra")
JANUARY = "Saturday nearest January 31"
CUSTOM = "Custom rules"


class ScopeQuestion(ValueError):
    def __init__(self, message: str, *, key: str = "", choices: tuple[str, ...] = ()):
        super().__init__(message)
        self.key = key
        self.choices = choices


def select_pattern(text, bindings):
    """Work on the masked question so quoted values and column names stay literal."""
    pattern = r"(?:4[-–]4[-–]5|4[-–]5[-–]4|5[-–]4[-–]4|NRF)"
    calendar_name = r"(?:(?:retail|fiscal)\s+)?calendar"
    def remember(token):
        token = token.upper().replace("–", "-")
        if token == "NRF":
            bindings.update(retail_pattern="4-5-4", retail_preset=JANUARY)
        else:
            bindings["retail_pattern"] = token

    def grouped(m):
        remember(m[1])
        return "by retail "
    text = re.sub(rf"\bby\s+({pattern})\s+(?=(?:month|period|week|quarter|year)s?\b)", grouped, text, flags=re.I)
    # Scope markings require a date expression or explicit calendar qualifier.
    for expr in (rf"\s*\(\s*({pattern})\s+{calendar_name}\s*\)",
                 rf"\s+(?:using|on|with)\s+(?:the |a |an )?({pattern})(?:\s+{calendar_name})?\b",
                 rf"\b({pattern})\s+(?=(?:retail\s+)?(?:fiscal|fy|period|week|quarter|[qpw]\d))",
                 rf"\b({pattern})\s+{calendar_name}\b"):
        def use(m):
            remember(m[1])
            return " "
        text = re.sub(expr, use, text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class RetailCalendar:
    pattern: str
    month: int
    weekday: int
    nearest: bool
    preceding_label: bool
    extra_period: int

    @classmethod
    def configured(cls, bindings):
        def choose(key, prompt, options):
            selected = bindings.get(key)
            if selected not in options:
                raise ScopeQuestion(prompt, key=key, choices=tuple(options))
            return selected
        pattern = choose("retail_pattern", "Which week pattern defines your retail periods?", PATTERNS)
        preset = choose("retail_preset",
            "Which year-end convention applies? The January preset ends on the Saturday nearest January 31, names FY for the preceding year, and puts the extra week in P12. Choose custom rules for another convention.",
            (JANUARY, CUSTOM))
        if preset == JANUARY:
            return cls(pattern, 1, 5, True, True, 12)
        month = choose("retail_month", "Which month-end anchors your retail year?", calendar.month_name[1:])
        day = choose("retail_weekday", "On which weekday does the retail year end?", calendar.day_name[:])
        rule = choose("retail_rule", "Use the weekday nearest month-end, or the last such weekday in the month?",
                      ("Nearest", "Last"))
        label = choose("retail_label", "Does the FY number name the year of the anchor month, or the year before it?",
                       ("Anchor year", "Preceding year"))
        extra = choose("retail_extra", "Which period receives the additional week in a 53-week year?",
                       ("P12", *(f"P{i}" for i in range(1, 12))))
        return cls(pattern, list(calendar.month_name).index(month), list(calendar.day_name).index(day),
                   rule == "Nearest", label == "Preceding year", int(extra[1:]))

    def end(self, anchor):
        last = date(anchor, self.month, calendar.monthrange(anchor, self.month)[1])
        back = (last.weekday() - self.weekday) % 7
        return last + timedelta(days=7-back if self.nearest and back > 3 else -back)

    def periods(self, anchor):
        start = self.end(anchor - 1) + timedelta(days=1)
        weeks = [int(n) for n in self.pattern.split("-")] * 4
        if (self.end(anchor) - self.end(anchor - 1)).days == 371:
            weeks[self.extra_period - 1] += 1
        bounds = [start]
        for n in weeks:
            bounds.append(bounds[-1] + timedelta(weeks=n))
        return bounds

    def containing(self, day):
        anchor = day.year
        while self.end(anchor) < day:
            anchor += 1
        while self.end(anchor - 1) >= day:
            anchor -= 1
        return anchor

    def interval(self, anchor, unit=None, number=None):
        bounds = self.periods(anchor)
        start, stop = bounds[0], bounds[-1]
        if unit == "P" and 1 <= number <= 12:
            start, stop = bounds[number - 1:number + 1]
        elif unit == "Q" and 1 <= number <= 4:
            start, stop = bounds[3*(number-1)], bounds[3*number]
        elif unit == "W" and 1 <= number <= (stop - start).days // 7:
            start = start + timedelta(weeks=number-1)
            stop = start + timedelta(weeks=1)
        elif unit:
            raise ScopeQuestion(f"Retail FY{anchor - int(self.preceding_label)} has no {unit}{number}.")
        return {"start": start.isoformat(), "end": stop.isoformat(), "retail_anchor": anchor,
                "retail_fy": anchor - int(self.preceding_label), "retail_pattern": self.pattern,
                "retail_unit": f"{unit}{number}" if unit else "Year",
                "retail_year_end": self.end(anchor).isoformat(), "extra_week_period": self.extra_period}

    def current(self, day, unit):
        anchor = self.containing(day)
        bounds = self.periods(anchor)
        period = next(i for i in range(1, 13) if day < bounds[i])
        number = {"W": (day-bounds[0]).days // 7 + 1, "P": period, "Q": (period-1)//3+1}.get(unit)
        return self.interval(anchor, unit, number)

    def buckets(self, low, high, unit):
        """Chronological labels and intervals, identical for pandas and disk aggregates."""
        for anchor in range(self.containing(low), self.containing(high) + 1):
            bounds = self.periods(anchor)
            if unit == "quarter":
                bounds = bounds[::3]
            elif unit == "year":
                bounds = [bounds[0], bounds[-1]]
            elif unit == "week":
                weeks = (bounds[-1] - bounds[0]).days // 7
                bounds = [bounds[0] + timedelta(weeks=i) for i in range(weeks + 1)]
            elif unit != "period":
                raise ValueError("Unknown retail grouping")
            for i, (start, stop) in enumerate(zip(bounds, bounds[1:]), 1):
                label = f"FY{anchor - int(self.preceding_label)}"
                if unit != "year":
                    label += f" {unit[0].upper()}{i:02}"
                yield start, stop, label


def retail_period(text, today, bindings):
    configured = bool(bindings.get("retail_pattern"))
    explicit = (re.search(r"\bretail\s+(?:fy|month|period|week|quarter|year|[qpw]\d)", text) or
                (re.search(r"\bfy\s*\d{4}", text) and re.search(r"\b(?:period|week|[pw])\s*\d{1,2}\b", text)))
    if not configured and not explicit:
        return None
    t = re.sub(r"\bretail month\b", "retail period", text)
    t = re.sub(r"\b(?:retail|fiscal)\s+", "", t).strip()
    t = re.sub(r"\b(?:period (?=p\d)|week (?=w\d)|quarter (?=q\d))", "", t)
    if re.fullmatch(r"(?:fytd|fqtd|year to date|quarter to date|period to date|week to date|(?:this|last|previous) (?:quarter|period|week|year))", t):
        cal = RetailCalendar.configured(bindings)
        unit = ("P" if "period" in t else "W" if "week" in t else
                "Q" if "quarter" in t or t == "fqtd" else None)
        part = cal.current(today, unit)
        if t.startswith(("last ", "previous ")):
            return cal.current(date.fromisoformat(part["start"]) - timedelta(days=1), unit)
        if t.endswith(("to date", "td")):
            part["end"] = (today + timedelta(days=1)).isoformat()
        return part
    if span := re.fullmatch(r"(?:between|from) (.+?) (to|through|until|~) (.+)", text):
        first = retail_period(span[1], today, bindings)
        if first is None:
            return None
        last = retail_period(span[3], today, {**bindings, "active_retail_anchor": first["retail_anchor"]})
        if last is None:
            raise ScopeQuestion("Give both endpoints as retail fiscal years, quarters, periods or weeks.")
        end = last["start"] if span[2] == "until" else last["end"]
        if end <= first["start"]:
            raise ScopeQuestion("The retail range ends before it starts. Give the FY for both endpoints.")
        return {**first, "end": end, "retail_last_anchor": last["retail_anchor"],
                "retail_unit": f"{first['retail_unit']} through {last['retail_unit']} FY{last['retail_fy']}"}
    t = re.sub(r"\b(period|week|quarter)\s*(\d+)\b", lambda m: m[1][0] + m[2], t)
    t = re.sub(r"\byear\s+(?=\d{4})", "fy", t)
    fy = r"fy\s*(\d{4}(?:[-/]\d{2,4})?)"
    subdivision = r"([qpw])\s*(\d{1,2})"
    if m := re.fullmatch(fy + r"(?:\s+" + subdivision + r")?", t):
        label, unit, n = m.groups()
    elif m := re.fullmatch(subdivision + r"(?:\s+(?:of\s+)?(?:" + fy + r"|(\d{4})))?", t):
        unit, n, year, plain = m.groups()
        label = year or plain
    else:
        return None
    cal = RetailCalendar.configured(bindings)
    label = label or bindings.get("fiscal_year")
    anchor = bindings.get("active_retail_anchor")
    if label:
        pieces = re.split(r"[-/]", label)
        y = int(pieces[0])
        if len(pieces) == 2:
            if pieces[1] not in {str(y+1), str(y+1)[-2:]}:
                raise ScopeQuestion("Give consecutive years for a two-year FY label.")
            anchor = y + 1
        else:
            anchor = y + int(cal.preceding_label)
    if anchor is None:
        raise ScopeQuestion("Which retail fiscal year should I use?", key="fiscal_year")
    return cal.interval(anchor, unit.upper() if unit else None, int(n) if n else None)
