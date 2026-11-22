"""Vehicle categories and generic model identification."""
from __future__ import annotations

from src.shared.numbers import norm

def category_for_model(model: object) -> str:
    t=set(norm(model).split())
    if t.intersection({'CAMINHAO','TRUCK','BASCULANTE','CAVALO','CARRETA'}): return 'caminhao'
    if t.intersection({'ONIBUS','MICRO','MARCOPOLO','COMIL','RODOVIARIO','ATHENAS','ALLEGRO'}): return 'onibus'
    if t.intersection({'PICK','PIC','S10','AMAROK','L200','FRONTIER','RANGER','HILUX','SAVEIRO','STRADA'}):return 'picape'
    if t.intersection({'VAN','SPRINTER','DUCATO','TRANSIT','JUMPER','MASTER'}): return 'van'
    if t.intersection({'REBOQUE','TANQUE'}): return 'outros'
    return 'automovel' if norm(model) else 'desconhecido'

def generic_model(value: object) -> bool:
    from .fields import GENERIC_EXACT,GENERIC_TOKENS
    n=norm(value)
    return n in GENERIC_EXACT or bool(set(n.split()) & GENERIC_TOKENS)

