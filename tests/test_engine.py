import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from fleet_force_engine.config import Settings,FLEET_COLUMNS
from fleet_force_engine.utils import norm
from fleet_force_engine.ingest import consolidate_raw_ufmg
from fleet_force_engine.deterministic import apply_quality,add_peer_benchmark,add_cost_analytics,summarize
from fleet_force_engine.learning import fit_efficiency,attach_efficiency_predictions
from fleet_force_engine.references import match_pbev


class EngineTests(unittest.TestCase):
    def sample_raw(self):
        rows=[]
        for idx,(kmpl,km) in enumerate([(8,2000),(10,2000),(11,2000),(12,2000)]):
            row=dict.fromkeys(FLEET_COLUMNS,0.0)
            row.update({'fleet_id':f'ABC{idx:04d}','model':'ONIX 1.0','km':km,'kmpl_reported':kmpl,
                        'total_expense':2000.,'fuel_expense':1500.,'os_count':1.})
            rows.append(row)
        return pd.DataFrame(rows)

    def test_same_model_leave_one_out_and_priority(self):
        fleet,_=consolidate_raw_ufmg(self.sample_raw())
        cfg=Settings()
        f=add_peer_benchmark(apply_quality(fleet,cfg),cfg)
        row=f.loc[f.fleet_id.eq('ABC0000')].iloc[0]
        self.assertAlmostEqual(row.peer_median_kmpl,11.)
        self.assertAlmostEqual(row.peer_overuse_pct,11/8-1)
        self.assertAlmostEqual(row.actionable_excess_liters_est,2000/8-2000/11)
        self.assertTrue(row.priority_alert)

    def test_duplicates_flagged_and_not_double_counted(self):
        raw=self.sample_raw()
        raw=pd.concat([raw,raw.iloc[[0]]],ignore_index=True)
        f,dupes=consolidate_raw_ufmg(raw)
        self.assertEqual(len(dupes),2)
        a=f.loc[f.fleet_id.eq('ABC0000')].iloc[0]
        self.assertEqual(a.km,2000)
        self.assertTrue(a.provisional_duplicate_vehicle)
        self.assertFalse(apply_quality(f,Settings()).loc[lambda x:x.fleet_id.eq('ABC0000'),'valid_efficiency'].iloc[0])

    def test_no_kmpl_when_missing(self):
        r=self.sample_raw();r.loc[0,'kmpl_reported']=0.
        f,_=consolidate_raw_ufmg(r)
        bad=f.loc[f.fleet_id.eq('ABC0000')].iloc[0]
        self.assertTrue(pd.isna(bad.kmpl))
        self.assertEqual(bad.km_uncovered,2000.)

    def test_category_and_pbev_not_for_truck(self):
        r=self.sample_raw();r['model']='CAMINHAO'
        f,_=consolidate_raw_ufmg(r)
        f=apply_quality(f,Settings())
        ref=pd.DataFrame([{'model_norm':'CAMINHAO','kmpl_reference':7.,'fuel':'DIESEL','reference_year':2026}])
        d=match_pbev(f,ref)
        self.assertTrue(d.pbev_min_kmpl.isna().all())

    def test_no_ml_promotion_without_external_period(self):
        cfg=Settings(ml_min_train_rows=2,ml_min_model_families=2)
        r=self.sample_raw()
        r['model']=['ONIX 1.0','CRONOS 1.3','LOGAN 1.6','FIESTA 1.6']
        f,_=consolidate_raw_ufmg(r);f=apply_quality(f,cfg)
        m,metrics,cv=fit_efficiency(f,cfg)
        self.assertFalse(metrics['approved_production'])

    def test_financial_cost_aggregation(self):
        raw=self.sample_raw();f,_=consolidate_raw_ufmg(raw)
        f=apply_quality(f,Settings());f=add_peer_benchmark(f,Settings());f=add_cost_analytics(f,Settings())
        kpis,categories=summarize(f)
        self.assertEqual(kpis['vehicles'],4)
        self.assertEqual(kpis['total_expense_brl'],8000)
        self.assertEqual(kpis['total_km'],8000)
        self.assertAlmostEqual(kpis['cost_per_km_brl'],1.)
        self.assertEqual(kpis['peer_priority_cases'],1)

if __name__=='__main__':unittest.main()
