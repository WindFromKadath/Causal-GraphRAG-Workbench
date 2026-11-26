import { useCallback, useEffect, useState } from "react";
import { FileUp, RefreshCw, RotateCcw, ShieldCheck, Trash2, Wrench } from "lucide-react";
import { apiClient } from "../api/client";
import type { DocumentItem, TenantSession } from "../types/api";

interface Props {
  session: TenantSession;
}

export function DocumentsWorkspace({ session }: Props) {
  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [dlqEntries, setDlqEntries] = useState<Array<Record<string, unknown>>>([]);
  const [loading, setLoading] = useState(false);
  const [banner, setBanner] = useState<{ kind: "ok" | "err"; text: string } | null>(null);

  const [docId, setDocId] = useState("doc_demo_manual");
  const [versionId, setVersionId] = useState(1);
  const [rawText, setRawText] = useState(
    "# 设备巡检补充规程\n\n## 1. 轴承温度告警\n离心泵滚动轴承温度超过 85°C 时，需立即核查润滑脂是否劣化或油位不足。\n\n## 2. 阈值表\n| 指标 | 告警 | 停机 |\n| --- | --- | --- |\n| 轴承温度 | 80°C | 95°C |",
  );

  const loadAll = useCallback(async () => {
    setLoading(true);
    try {
      const [dRes, dlqRes] = await Promise.all([apiClient.listDocuments(session, true), apiClient.listDlq(session)]);
      setDocs(dRes.documents);
      setDlqEntries(dlqRes.entries);
    } catch (err) {
      setBanner({ kind: "err", text: err instanceof Error ? err.message : String(err) });
    } finally {
      setLoading(false);
    }
  }, [session]);

  useEffect(() => {
    void loadAll();
  }, [loadAll]);

  async function handleFileUpload(file: File | undefined) {
    if (!file) return;
    setBanner(null);
    try {
      const res = await apiClient.uploadFile(session, file, undefined, 1, session.aclRoles);
      setBanner({
        kind: "ok",
        text: `上传入库成功：${JSON.stringify(res)}`,
      });
      await loadAll();
    } catch (err) {
      setBanner({ kind: "err", text: `质检拦截 / 上传失败：${err instanceof Error ? err.message : String(err)}` });
    }
  }

  async function handleInlineIngest() {
    setBanner(null);
    try {
      const res = await apiClient.ingestText(session, {
        doc_id: docId,
        title: docId,
        raw_text: rawText,
        version_id: versionId,
        via_redis: false,
      });
      setBanner({
        kind: "ok",
        text: `增量入库结果：outcome=${String(res.outcome)} | reused=${String(res.reused_embeddings ?? 0)} | computed=${String(res.computed_embeddings ?? 0)}`,
      });
      await loadAll();
    } catch (err) {
      setBanner({ kind: "err", text: err instanceof Error ? err.message : String(err) });
    }
  }

  async function handleRollback(id: string, currentVer: number) {
    const target = Math.max(1, currentVer - 1);
    setBanner(null);
    try {
      const res = await apiClient.rollbackDocument(session, id, target);
      setBanner({ kind: "ok", text: `回滚成功：${JSON.stringify(res)}` });
      await loadAll();
    } catch (err) {
      setBanner({ kind: "err", text: err instanceof Error ? err.message : String(err) });
    }
  }

  async function handleDelete(id: string) {
    setBanner(null);
    try {
      const res = await apiClient.deleteDocument(session, id);
      setBanner({ kind: "ok", text: `双库同步软删除完成：${JSON.stringify(res)}` });
      await loadAll();
    } catch (err) {
      setBanner({ kind: "err", text: err instanceof Error ? err.message : String(err) });
    }
  }

  async function handleReconcile() {
    setBanner(null);
    try {
      const res = await apiClient.reconcile(session, 300, false);
      setBanner({ kind: "ok", text: `对账完成：${JSON.stringify(res)}` });
      await loadAll();
    } catch (err) {
      setBanner({ kind: "err", text: err instanceof Error ? err.message : String(err) });
    }
  }

  return (
    <div className="space-y-4">
      {banner && (
        <div
          className={`rounded-lg border px-4 py-2.5 text-xs ${
            banner.kind === "ok"
              ? "border-emerald-500/50 bg-emerald-500/10 text-emerald-300"
              : "border-rose-500/50 bg-rose-500/10 text-rose-300"
          }`}
        >
          {banner.text}
        </div>
      )}

      {/* Upload + Inline Ingest */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="panel p-4 space-y-3">
          <h3 className="text-sm font-semibold flex items-center gap-2">
            <FileUp className="w-4 h-4 text-sky-400" />
            多格式文件上传（PDF / DOCX / MD / PY + 三道质检）
          </h3>
          <label className="block border-2 border-dashed border-slate-600 hover:border-sky-500 rounded-xl p-6 text-center cursor-pointer transition-colors">
            <input
              type="file"
              className="hidden"
              accept=".pdf,.docx,.md,.txt,.py"
              onChange={(e) => void handleFileUpload(e.target.files?.[0])}
            />
            <p className="text-sm font-medium">点击选择或拖拽文件上传</p>
            <p className="text-xs text-slate-400 mt-1">
              自动执行 FileGate（MIME 防伪造）→ ParseGate（乱码率/缺页检测）→ Parent-Child 分块 → ChunkGate
            </p>
          </label>
          <div className="flex flex-wrap gap-2 pt-1">
            <button
              onClick={() => void handleReconcile()}
              className="px-3 py-1.5 rounded-lg panel-subtle hover:border-sky-500 text-xs flex items-center gap-1.5 cursor-pointer"
            >
              <Wrench className="w-3.5 h-3.5 text-amber-400" />
              执行对账（Reconcile）
            </button>
            <button
              onClick={() => void loadAll()}
              className="px-3 py-1.5 rounded-lg panel-subtle hover:border-sky-500 text-xs flex items-center gap-1.5 cursor-pointer"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
              刷新列表
            </button>
          </div>
        </div>

        <div className="panel p-4 space-y-2.5">
          <h3 className="text-sm font-semibold flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            增量版本入库测试（SHA-256 未变块零重算 Embedding）
          </h3>
          <div className="flex gap-2">
            <input
              value={docId}
              onChange={(e) => setDocId(e.target.value)}
              className="panel-subtle px-2.5 py-1.5 text-xs flex-1 outline-none"
              placeholder="doc_id"
            />
            <input
              type="number"
              min={1}
              value={versionId}
              onChange={(e) => setVersionId(Number(e.target.value) || 1)}
              className="panel-subtle px-2.5 py-1.5 text-xs w-24 outline-none"
            />
            <button
              onClick={() => void handleInlineIngest()}
              className="px-4 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-500 text-white text-xs font-medium cursor-pointer"
            >
              提交 v{versionId}
            </button>
          </div>
          <textarea
            value={rawText}
            onChange={(e) => setRawText(e.target.value)}
            rows={5}
            className="panel-subtle w-full px-3 py-2 text-xs font-mono outline-none"
          />
        </div>
      </div>

      {/* Documents Table */}
      <div className="panel p-4">
        <h3 className="text-sm font-semibold mb-3">
          租户文档与版本账本（<code>{session.tenantId}</code> · 共 {docs.length} 份）
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="border-b border-slate-700 text-slate-400">
              <tr>
                <th className="py-2 pr-3">Doc ID</th>
                <th className="py-2 pr-3">标题 / 来源</th>
                <th className="py-2 pr-3">Active Ver</th>
                <th className="py-2 pr-3">状态</th>
                <th className="py-2 pr-3">切片数</th>
                <th className="py-2 pr-3">ACL</th>
                <th className="py-2 text-right">操作（需 admin）</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {docs.map((d) => (
                <tr key={d.doc_id} className={d.is_deleted ? "opacity-50" : ""}>
                  <td className="py-2 pr-3 font-mono text-sky-400">{d.doc_id}</td>
                  <td className="py-2 pr-3">
                    <div className="font-medium">{d.title}</div>
                    <div className="text-[11px] text-slate-400">{d.source_uri}</div>
                  </td>
                  <td className="py-2 pr-3 font-mono">v{d.active_version}</td>
                  <td className="py-2 pr-3">
                    <span
                      className={`px-2 py-0.5 rounded-full text-[11px] ${
                        d.is_deleted
                          ? "bg-rose-500/20 text-rose-300"
                          : d.status === "ready"
                            ? "bg-emerald-500/20 text-emerald-300"
                            : "bg-amber-500/20 text-amber-300"
                      }`}
                    >
                      {d.is_deleted ? "deleted" : d.status}
                    </span>
                  </td>
                  <td className="py-2 pr-3">{d.chunk_count}</td>
                  <td className="py-2 pr-3 font-mono text-[11px]">{d.acl.join(", ")}</td>
                  <td className="py-2 text-right space-x-2">
                    {d.active_version > 1 && !d.is_deleted && (
                      <button
                        onClick={() => void handleRollback(d.doc_id, d.active_version)}
                        className="px-2 py-1 rounded bg-amber-500/15 hover:bg-amber-500/25 text-amber-300 inline-flex items-center gap-1 cursor-pointer"
                      >
                        <RotateCcw className="w-3 h-3" /> 回滚
                      </button>
                    )}
                    {!d.is_deleted && (
                      <button
                        onClick={() => void handleDelete(d.doc_id)}
                        className="px-2 py-1 rounded bg-rose-500/15 hover:bg-rose-500/25 text-rose-300 inline-flex items-center gap-1 cursor-pointer"
                      >
                        <Trash2 className="w-3 h-3" /> 软删
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* DLQ Panel */}
      <div className="panel p-4">
        <h3 className="text-sm font-semibold mb-2">Redis 死信队列（DLQ · {dlqEntries.length} 条）</h3>
        {dlqEntries.length === 0 ? (
          <p className="text-xs text-emerald-400">当前无失败死信事件（0 DLQ）。</p>
        ) : (
          <pre className="panel-subtle p-3 text-xs overflow-x-auto">{JSON.stringify(dlqEntries, null, 2)}</pre>
        )}
      </div>
    </div>
  );
}
