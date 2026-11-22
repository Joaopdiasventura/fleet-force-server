"""CLI de execução do único motor analítico.

Exemplo:
  python run.py --fleet-pdf 'despesa-veiculos-ufmg-2025.pdf' \
    --anp-monthly 'data/anp_precos_mensais_uf.csv' \
    --pbev-csv 'data/pbev_catalogo_historico_extraido_auditar.csv'
"""
import argparse
import json
from pathlib import Path
from src.config.engine import run_engine
from src.core.policy import Settings
from src.shared.logging import configure_logging


def choose(*paths: str | None) -> str | None:
    for p in paths:
        if p and Path(p).is_file():return str(p)
    return None

def main() -> None:
    configure_logging()
    p=argparse.ArgumentParser(description='Fleet Force - motor analítico determinístico com ML validado')
    p.add_argument('--fleet-pdf',help='PDF UFMG ou relatório com layout equivalente')
    p.add_argument('--fleet-csv',help='CSV de frota agregada (preferível se gerado pelo main.ipynb)')
    p.add_argument('--source-zip',help='ZIP original ANP/PBEV (processamento completo se sem caches)')
    p.add_argument('--anp-monthly',help='CSV mensal ANP já agregado')
    p.add_argument('--pbev-csv',help='CSV de PBEV normalizado e extraído')
    p.add_argument('--external-validation',help='CSV de outro período para validação realmente independente de ML UFMG')
    p.add_argument('--uf',default='MG',help='UF usada nos cenários de preços (default: MG)')
    p.add_argument('--out',default='outputs',help='Diretório de exportações')
    p.add_argument('--rebuild-references',action='store_true',help='Reprocessa ZIP mesmo se há CSV cache')
    args=p.parse_args()
    csv=choose(args.fleet_csv,'fleet_analysis_outputs/fleet_vehicles.csv') if not args.fleet_pdf else args.fleet_csv
    pdf=choose(args.fleet_pdf,'data/despesa-veiculos-ufmg-2025.pdf','despesa-veiculos-ufmg-2025.pdf','despesa-veiculos-ufmg-2025(1).pdf') if not csv else args.fleet_pdf
    z=choose(args.source_zip,'fleet-force.zip')
    anp=choose(args.anp_monthly,'data/anp_precos_mensais_uf.csv')
    pbev=choose(args.pbev_csv,'data/pbev_catalogo_historico_extraido_auditar.csv')
    if args.rebuild_references and not z:p.error('--rebuild-references exige --source-zip (arquivo ZIP original).')
    if not pdf and not csv:p.error('Frota ausente: informe --fleet-pdf ou --fleet-csv')
    result=run_engine(fleet_pdf=pdf,fleet_csv=csv,source_zip=z,anp_monthly_csv=anp,
                      pbev_csv=pbev,external_validation_csv=args.external_validation,
                      cfg=Settings(price_uf=args.uf.upper()),output_dir=args.out,
                      rebuild_references=args.rebuild_references)
    print(json.dumps({'kpis':result.kpis,
                      'ml_efficiency_approved':result.training_metrics['efficiency']['approved_production'],
                      'ml_price_approved':result.training_metrics['price']['approved_production'],
                      'outputs':str(Path(args.out).resolve())},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
