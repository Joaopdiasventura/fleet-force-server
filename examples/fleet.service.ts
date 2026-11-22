// Exemplo Angular 17+ / 21: interface REST sem lógica estatística no frontend.
// Ajuste baseUrl para a sua API e adicione autenticação/autorização reais.
import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface FleetSummary {
  vehicles: number;
  eligible_vehicles: number;
  quality_review_vehicles: number;
  total_km: number;
  total_expense_brl: number;
  cost_per_km_brl: number | null;
  peer_priority_cases: number;
  unified_priority_cases: number;
  validated_ml_reference_vehicles: number;
}

export type ReferenceSource =
  | 'deterministic_same_model_peer'
  | 'ml_approved_external_validation'
  | 'none';

export interface VehicleInsight {
  fleet_id: string;
  model: string;
  vehicle_category: string;
  km: number;
  kmpl: number | null;
  total_expense: number | null;
  cost_per_km: number | null;
  peer_confidence: 'moderate' | 'low' | 'insufficient';
  priority_alert: boolean;
  ml_priority_alert: boolean;
  active_reference_source: ReferenceSource;
  active_reference_kmpl: number | null;
  unified_priority_liters_est: number | null;
  quality_reason: string | null;
}

export interface Page<T> {
  total: number;
  offset: number;
  limit: number;
  items: T[];
}

@Injectable({ providedIn: 'root' })
export class FleetAnalyticsService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = '/api/v1/fleet';

  summary(): Observable<FleetSummary> {
    return this.http.get<FleetSummary>(`${this.baseUrl}/summary`);
  }

  vehicles(offset = 0, limit = 50, onlyPriorities = false): Observable<Page<VehicleInsight>> {
    const params = new HttpParams()
      .set('offset', offset)
      .set('limit', limit)
      .set('only_priorities', onlyPriorities);
    return this.http.get<Page<VehicleInsight>>(`${this.baseUrl}/vehicles`, { params });
  }

  vehicle(id: string): Observable<VehicleInsight> {
    return this.http.get<VehicleInsight>(`${this.baseUrl}/vehicles/${encodeURIComponent(id)}`);
  }

  quality(): Observable<Record<string, unknown>> {
    return this.http.get<Record<string, unknown>>(`${this.baseUrl}/models/quality`);
  }
}
