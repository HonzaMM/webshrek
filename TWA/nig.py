"""
Jednoduchý AI agent (Claude API + nástroje + smyčka).

Spuštění:
    pip install anthropic
    export ANTHROPIC_API_KEY="tvuj-klic"      # Windows: set ANTHROPIC_API_KEY=tvuj-klic
    python agent.py
"""
import ast
import operator
import re
import urllib.request
from datetime import datetime
from pathlib import Path

import anthropic

MODEL = "claude-sonnet-5-5"
MAX_KROKU = 10  # pojistka proti nekonečné smyčce
PRACOVNI_SLOZKA = Path("workspace").resolve()  # agent smí jen sem
PRACOVNI_SLOZKA.mkdir(exist_ok=True)

SYSTEM = (
    "Jsi užitečný asistent běžící jako agent s nástroji. Odpovídej česky, stručně a věcně. "
    "Když potřebuješ data, použij nástroj, nehádej. Soubory můžeš číst a psát jen ve své "
    "pracovní složce. Když nástroj selže, zkus to jinak nebo to poctivě řekni."
)

client = anthropic.Anthropic()  # klíč se bere z ANTHROPIC_API_KEY


# ---------- NÁSTROJE (obyčejné pythonní funkce) ----------

def bezpecna_cesta(cesta: str) -> Path:
    p = (PRACOVNI_SLOZKA / cesta).resolve()
    if not p.is_relative_to(PRACOVNI_SLOZKA):
        raise ValueError("Cesta mimo pracovní složku není povolená.")
    return p


def vypis_slozku(cesta: str = ".") -> str:
    p = bezpecna_cesta(cesta)
    polozky = sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())
    return "\n".join(polozky) or "(prázdná složka)"


def precti_soubor(cesta: str) -> str:
    return bezpecna_cesta(cesta).read_text(encoding="utf-8")[:20000]


def zapis_soubor(cesta: str, obsah: str) -> str:
    # Pojistka: nebezpečná akce vyžaduje potvrzení člověkem
    odp = input(f"\n⚠️  Agent chce zapsat do '{cesta}' ({len(obsah)} znaků). Povolit? [a/N] ")
    if odp.strip().lower() != "a":
        return "Uživatel zápis zamítl."
    p = bezpecna_cesta(cesta)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(obsah, encoding="utf-8")
    return f"Zapsáno do {cesta}."


def stahni_stranku(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return "Povolené jsou jen http(s) adresy."
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (moje-agent)"})
    with urllib.request.urlopen(req, timeout=15) as r:
        html = r.read(500_000).decode("utf-8", errors="ignore")
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()[:8000]


OPERATORY = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def vypocet(vyraz: str) -> str:
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in OPERATORY:
            return OPERATORY[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in OPERATORY:
            return OPERATORY[type(n.op)](ev(n.operand))
        raise ValueError("Nepovolený výraz.")
    return str(ev(ast.parse(vyraz, mode="eval").body))


def aktualni_cas() -> str:
    return datetime.now().strftime("%A %d.%m.%Y %H:%M:%S")


FUNKCE = {
    "vypis_slozku": vypis_slozku,
    "precti_soubor": precti_soubor,
    "zapis_soubor": zapis_soubor,
    "stahni_stranku": stahni_stranku,
    "vypocet": vypocet,
    "aktualni_cas": aktualni_cas,
}

# Popis nástrojů pro model (ten podle nich rozhoduje, co zavolat)
TOOLS = [
    {"name": "vypis_slozku", "description": "Vypíše obsah složky v pracovní složce.",
     "input_schema": {"type": "object", "properties": {"cesta": {"type": "string"}}}},
    {"name": "precti_soubor", "description": "Přečte textový soubor z pracovní složky.",
     "input_schema": {"type": "object", "properties": {"cesta": {"type": "string"}},
                      "required": ["cesta"]}},
    {"name": "zapis_soubor", "description": "Zapíše text do souboru v pracovní složce (vyžaduje potvrzení uživatele).",
     "input_schema": {"type": "object",
                      "properties": {"cesta": {"type": "string"}, "obsah": {"type": "string"}},
                      "required": ["cesta", "obsah"]}},
    {"name": "stahni_stranku", "description": "Stáhne webovou stránku a vrátí její text.",
     "input_schema": {"type": "object", "properties": {"url": {"type": "string"}},
                      "required": ["url"]}},
    {"name": "vypocet", "description": "Spočítá matematický výraz, např. '(12+3)*4/5'.",
     "input_schema": {"type": "object", "properties": {"vyraz": {"type": "string"}},
                      "required": ["vyraz"]}},
    {"name": "aktualni_cas", "description": "Vrátí aktuální datum a čas.",
     "input_schema": {"type": "object", "properties": {}}},
]


# ---------- SMYČKA AGENTA ----------

def spust_agenta(messages: list) -> None:
    for krok in range(1, MAX_KROKU + 1):
        r = client.messages.create(
            model=MODEL, max_tokens=2000, system=SYSTEM, tools=TOOLS, messages=messages
        )
        messages.append({"role": "assistant", "content": r.content})

        if r.stop_reason != "tool_use":  # model odpověděl textem -> hotovo
            print("\n🤖", "".join(b.text for b in r.content if b.type == "text"))
            return

        vysledky = []
        for b in r.content:
            if b.type != "tool_use":
                continue
            print(f"  🔧 [{krok}] {b.name}({b.input})")
            try:
                vystup, chyba = str(FUNKCE[b.name](**b.input)), False
            except Exception as e:
                vystup, chyba = f"Chyba: {e}", True
            vysledky.append({"type": "tool_result", "tool_use_id": b.id,
                             "content": vystup, "is_error": chyba})
        messages.append({"role": "user", "content": vysledky})

    raise RuntimeError(f"Agent nedokončil úkol za {MAX_KROKU} kroků.")


def main() -> None:
    print("Agent připraven. Pracovní složka:", PRACOVNI_SLOZKA)
    print("Napiš úkol (prázdný řádek = konec).")
    messages: list = []
    while True:
        zadani = input("\n👤 ").strip()
        if not zadani:
            break
        zacatek = len(messages)  # kdyby se něco pokazilo, vrátíme historii zpět
        messages.append({"role": "user", "content": zadani})
        try:
            spust_agenta(messages)
        except Exception as e:
            print(f"\n❌ {e}")
            del messages[zacatek:]


if __name__ == "__main__":
    main()