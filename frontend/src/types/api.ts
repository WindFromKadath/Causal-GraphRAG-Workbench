export interface TenantSession {
  tenantId: string;
  aclRoles: string;
  operator: string;
  apiKey: string;
}

export interface ServiceHealth {
  status: string;
  vector_backend: string;
  graph_backend: string;
  checkpointer: string;
  llm_model: string;
  embedding_model: string;
  reranker_backend: string;
}

export interface InfraServiceItem {
  service: string;
  ok: boolean;
  endpoint: string;
  detail: string;
}

export interface InfraHealthResponse {
  all_healthy: boolean;
  services: InfraServiceItem[];
}

export interface WaterfallStep {
  node: string;
  duration_ms: number;
  status: string;
  summary: Record<string, unknown>;
}

export interface RerankedHit {
  chunk_id: string;
  doc_id: string;
  coarse_rank: number;
  final_rank: number;
  rerank_score: number;
  snippet: string;
}

export interface TotBranch {
  branch_id: string;
  status: string;
  score: number;
  deterministic_graph_score?: number;
  semantic_score?: number;
  hypothesis: string;
  target_entity_name?: string;
  target_entity_id?: string;
  prune_reason?: string;
}

export interface CotStep {
  step_id: number;
  stage: string;
  grounded: boolean;
  claim: string;
  cited_entity_ids?: string[];
  cited_relation_ids?: string[];
  cited_chunk_ids?: string[];
}

export interface AgentRunOutput {
  request_id: string;
  thread_id: string;
  tenant_id: string;
  acl_roles: string[];
  question: string;
  status: "completed" | "interrupted_hitl" | "error";
  answer: string;
  rewritten_queries: string[];
  reranker_backend: string;
  reranked_hits: RerankedHit[];
  evidence_score: number;
  retry_count: number;
  grounding_retry_count: number;
  grounding_score: number | null;
  route_decision: Record<string, unknown>;
  interrupt_payload: Record<string, unknown> | null;
  writeback_decision: Record<string, unknown> | null;
  waterfall: WaterfallStep[];
  total_ms: number;
  query_result: {
    metadata?: {
      reasoning_paradigm?: string;
      matched_hyperedges?: Array<{
        matched_condition_names?: string[];
        target_entity_name?: string;
        coverage_ratio?: number;
      }>;
      recalled_memories?: Array<{ memory_id?: string; conclusion?: string }>;
      reasoning?: {
        confidence?: number;
        branches?: TotBranch[];
        steps?: CotStep[];
        grounding_audit?: {
          grounding_score: number;
          grounded_steps: number;
          total_steps: number;
          passed: boolean;
          ungrounded_citations: string[];
        };
      };
    };
  };
}

export interface SseEventPacket {
  event: string;
  node: string;
  thread_id?: string;
  request_id?: string;
  data: Record<string, unknown>;
}

export interface DocumentItem {
  doc_id: string;
  title: string;
  source_type: string;
  source_uri: string;
  active_version: number;
  status: string;
  is_deleted: boolean;
  acl: string[];
  missing_pages: number[];
  updated_at?: string;
  chunk_count: number;
}

export interface EntityItem {
  entity_id: string;
  name: string;
  entity_type: string;
  description: string;
  aliases: string[];
  source_chunk_ids: string[];
  community_id: number | null;
  metadata?: Record<string, unknown>;
}

export interface RelationItem {
  relation_id: string;
  source_id: string;
  target_id: string;
  relation_type: string;
  description: string;
  weight: number;
  source_chunk_ids: string[];
}

export interface HyperEdgeItem {
  hyperedge_id: string;
  condition_entity_ids: string[];
  target_entity_id: string;
  relation_type: string;
  description: string;
  confidence: number;
  weight: number;
  source_chunk_ids: string[];
}

export interface CommunityItem {
  community_id: number;
  title: string;
  summary: string;
  entity_ids: string[];
  relation_count: number;
}

export interface GraphOverviewResponse {
  tenant_id: string;
  stats: {
    entity_count: number;
    relation_count: number;
    hyperedge_count: number;
    community_count: number;
  };
  entities: EntityItem[];
  relations: RelationItem[];
  hyperedges: HyperEdgeItem[];
  communities: CommunityItem[];
}

export interface ShaclCheckItem {
  check_id: string;
  title: string;
  severity: string;
  passed: boolean;
  violation_count: number;
  details?: string;
}

export interface ShaclReportResponse {
  profile: string;
  overall_status: string;
  compliance_score: number;
  checks: ShaclCheckItem[];
}

export interface VerifiedMemoryItem {
  memory_id: string;
  question: string;
  conclusion: string;
  condition_entity_ids: string[];
  target_entity_ids: string[];
  source_chunk_ids: string[];
  grounding_score: number;
  confidence: number;
  reasoning_paradigm: string;
  valid_from: string;
  provenance_run_id: string;
  status: string;
}

export interface RequestTraceItem {
  request_id: string;
  thread_id: string;
  tenant_id: string;
  acl: string[];
  question: string;
  rewritten_queries: string[];
  route_decision: Record<string, unknown>;
  coarse_hits: Array<{ chunk_id: string; score: number }>;
  reranked_hits: RerankedHit[];
  grounding_score: number | null;
  terminal_status: string;
  waterfall: WaterfallStep[];
  total_ms: number;
  model_versions: Record<string, string>;
  created_at: string;
}

export interface EvalMetricItemInput {
  question_id: string;
  question_type: string;
  prediction: string;
  answer_accuracy: number;
  factuality_f1: number;
  semantic_similarity: number;
  faithfulness: number;
  semantic_coverage: number;
  retrieved_ids: string[];
  gold_ids: string[];
  execution_error: boolean;
}

export interface EvalSummaryResponse {
  summary: {
    mode: string;
    total: number;
    valid_scored_count: number;
    execution_error_count: number;
    avg_answer_accuracy: number;
    avg_faithfulness: number;
    avg_semantic_coverage: number;
    refusal_rate: number;
    business_refusal_rate: number;
    semantic_refusal_rate: number;
    avg_hit_at_5: number | null;
    avg_mrr_at_10: number | null;
    answerability_distribution: Record<string, number>;
  };
  questions: Array<{
    question_id: string;
    question_type: string;
    answer_accuracy: number;
    faithfulness: number;
    semantic_coverage: number;
    answerability: string;
    refusal_kind: string;
    hit_at_5: number | null;
    mrr_at_10: number | null;
  }>;
}
