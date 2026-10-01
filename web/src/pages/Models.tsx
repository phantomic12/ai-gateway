import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import {
  Badge, Button, Card, Confirm, CopyButton, EmptyState, Field, Icon, Input,
  PageHeader, Select, Skeleton, useToast,
} from '../components/ui'

function parseLevels(s?: string): string[] {
  if(!s) return []
  try { const v=JSON.parse(s); return Array.isArray(v)? v: [] } catch { return s.split(',').map(x=>x.trim()).filter(Boolean) }
}
function parseLimits(s?: string): Record<string, number> {
  if(!s) return {}
  try { return JSON.parse(s) } catch { return {} }
}

// Human-sized token windows: 1000k -> 1M, 1048576 -> 1.05M, 131072 -> 131k.
function fmtK(n?: number): string {
  if(!n) return '-'
  if(n >= 1000000){ const mv = n/1000000; return (mv % 1 === 0 ? mv.toFixed(0) : String(parseFloat(mv.toFixed(2)))) + 'M' }
  return Math.round(n/1000) + 'k'
}
// Prices are per 1M tokens; keep short so two prices never wrap in a card cell.
function fmtCost(v?: number): string {
  if(v == null) return '-'
  if(v >= 100) return v.toFixed(0)
  if(v >= 10) return v.toFixed(1)
  return v.toFixed(2)
}
function fmtMs(ms: number): string {
  if(ms < 1000) return Math.round(ms) + 'ms'
  if(ms < 60000) return (ms/1000).toFixed(1) + 's'
  return Math.round(ms/1000) + 's'
}

type PendingDelete =
  | { kind:'bulk' }
  | { kind:'model'; id:string; label:string }
  | { kind:'alias'; alias:string }

function sourceTone(source?: string): 'neutral'|'good'|'warn' {
  if(source==='manual') return 'warn'
  if(source==='provider'||source==='enriched') return 'good'
  return 'neutral'
}

export default function Models({ role = 'admin' }: { role?: string }){
  // Provider-model mutations (add/edit/enrich/delete/discover/sync/aliases)
  // are admin-only server-side.
  const isAdmin = role === 'admin'
  const [providers, setProviders] = useState<any[]>([])
  const [list, setList] = useState<any[]>([])
  const [q, setQ] = useState('')
  const [provider, setProvider] = useState('')
  const [status, setStatus] = useState<any>(null)
  const [syncing, setSyncing] = useState(false)
  const [discovering, setDiscovering] = useState(false)
  const [aliases, setAliases] = useState<any[]>([])
  const [newAlias, setNewAlias] = useState('')
  const [newTarget, setNewTarget] = useState('')
  const [showAdd, setShowAdd] = useState(false)
  const [addProvider, setAddProvider] = useState('')
  const [addModel, setAddModel] = useState('')
  const [editId, setEditId] = useState<string|null>(null)
  const [edit, setEdit] = useState<any>({})
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [busy, setBusy] = useState(false)

  // Presentation-only: initial skeleton grid + destructive confirm modal.
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState('')
  const [pendingDelete, setPendingDelete] = useState<PendingDelete|null>(null)
  const [excluded, setExcluded] = useState<any[]>([])
  const [binOpen, setBinOpen] = useState(false)
  const [restoring, setRestoring] = useState<string|null>(null)
  const toast = useToast()

  const load = async()=>{
    try{
      const data = await api.providerModels.list(provider || undefined, q || undefined)
      setList(data.data || [])
      setLoadError('')
    }catch(e:any){ setLoadError(e?.message || String(e)) }
    api.catalog.status().then(setStatus).catch(()=>{})
    api.catalog.aliases().then(setAliases).catch(()=>{})
    api.providers.list().then(setProviders).catch(()=>{})
    api.providerModels.excluded(provider || undefined, q || undefined).then(d=>{
      const rows = d.data || []
      setExcluded(rows)
      if(q && rows.length) setBinOpen(true)
    }).catch(()=>{})
    setLoading(false)
  }
  useEffect(()=>{ load()},[])
  useEffect(()=>{ const t=setTimeout(load, 300); return ()=>clearTimeout(t)},[q, provider])

  const syncCatalog = async()=>{
    setSyncing(true)
    try{ await api.catalog.sync(); await load(); toast.success('Catalog synced') }catch(e:any){ toast.error(e.message || String(e)) } finally{ setSyncing(false)}
  }
  const discover = async()=>{
    setDiscovering(true)
    try{
      if(provider){ await api.providerModels.discover(provider) } else { await api.providerModels.discoverAll() }
      await load()
      toast.success('Discovery complete')
    }catch(e:any){ toast.error(e.message || String(e))} finally{ setDiscovering(false)}
  }

  const toggleSelect = (id:string)=>{
    setSelected(prev=>{
      const next = new Set(prev)
      if(next.has(id)) next.delete(id); else next.add(id)
      return next
    })
  }
  const toggleAll = ()=>{
    if(selected.size===list.length) setSelected(new Set())
    else setSelected(new Set(list.map(m=>m.id)))
  }
  const clearSelection = ()=> setSelected(new Set())
  const isSelectMode = selected.size>0

  const bulkEnrich = async()=>{
    if(selected.size===0) return
    setBusy(true)
    try{ await api.providerModels.bulkEnrich(Array.from(selected)); await load(); clearSelection(); toast.success(`Enriched ${selected.size} model(s)`) }catch(e:any){ toast.error(e.message || String(e))} finally{ setBusy(false)}
  }
  // Confirm-gated by pendingDelete {kind:'bulk'}.
  const bulkDelete = async()=>{
    if(selected.size===0) return
    setBusy(true)
    try{ await api.providerModels.bulkRemove(Array.from(selected)); await load(); clearSelection(); toast.success('Moved to the recycling bin — they will not be auto-discovered again') }catch(e:any){ toast.error(e.message || String(e))} finally{ setBusy(false)}
  }

  const confirmDelete = ()=>{
    if(!pendingDelete) return
    if(pendingDelete.kind==='bulk') bulkDelete()
    else if(pendingDelete.kind==='model'){
      api.providerModels.remove(pendingDelete.id)
        .then(()=>{ toast.success('Model moved to the recycling bin — it will not be auto-discovered again'); return load() })
        .catch((e:any)=> toast.error(e.message || String(e)))
        .finally(()=> setPendingDelete(null))
    } else {
      api.catalog.deleteAlias(pendingDelete.alias)
        .then(()=>{ toast.success('Alias removed'); return load() })
        .catch((e:any)=> toast.error(e.message || String(e)))
        .finally(()=> setPendingDelete(null))
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Connect · Catalog"
        title="Models"
        description="Catalog and per-provider models with context windows, pricing and live traffic. Search, enrich, and alias." 
        actions={
          isAdmin ? (
            <>
              <Button variant="secondary" onClick={syncCatalog} disabled={syncing}>
                <Icon name="refresh" size={14} className={syncing?'animate-spin':''}/> {syncing?'Syncing':'Sync'}
              </Button>
              <Button variant="primary" onClick={discover} disabled={discovering || providers.length===0}>
                <Icon name="zap" size={14}/> {discovering?'Discovering':'Discover'}
              </Button>
            </>
          ) : undefined
        }
      />

      {loadError && (
        <Card>
          <EmptyState
            icon="alert"
            title="Could not load models"
            hint={loadError}
            action={<Button variant="secondary" onClick={load}><Icon name="refresh" size={15}/>Retry</Button>}
          />
        </Card>
      )}

      {status && <div className="font-mono text-xs text-muted">{status.count} catalog · {list.length} provider models{isAdmin && isSelectMode && <> · <span className="text-amber">{selected.size} selected</span></>}</div>}
      {providers.length===0 && (
        <div className="border border-amber/30 bg-amber/10 rounded-xl p-3 text-sm flex items-center gap-2">
          <Icon name="alert" size={14} className="text-amber shrink-0"/> Add a provider first{isAdmin ? ', then Discover.' : ' (admin-only).'}
        </div>
      )}

      {/* Search / filter row */}
      <Card className="flex flex-col sm:flex-row flex-wrap items-stretch sm:items-center gap-2 !rounded-xl !p-3">
        {isAdmin && (
          <label className="flex items-center gap-1.5 text-xs text-muted shrink-0 px-1 cursor-pointer">
            <input type="checkbox" checked={list.length>0 && selected.size===list.length} onChange={toggleAll} aria-label="Select all models" className="accent-accent rounded"/>
            All
          </label>
        )}
        <Select value={provider} onChange={e=>setProvider(e.target.value)} className="sm:w-48">
          <option value="">All providers</option>
          {providers.map((p:any)=> <option key={p.id} value={p.id}>{p.name}</option>)}
        </Select>
        <div className="relative flex-1 min-w-[180px]">
          <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted pointer-events-none"><Icon name="search" size={14}/></span>
          <Input value={q} onChange={e=>setQ(e.target.value)} placeholder="Search model" className="pl-9"/>
        </div>
        {isAdmin && (
          <Button variant="ghost" size="md" onClick={()=>setShowAdd(!showAdd)}>
            <Icon name="plus" size={15}/> Add
          </Button>
        )}
        <span className="font-mono text-xs text-muted self-center px-1">{list.length}</span>
      </Card>

      {isAdmin && isSelectMode && (
        <div className="border border-amber/30 bg-amber/10 rounded-xl p-3 flex flex-wrap items-center justify-between gap-3">
          <Badge tone="warn">{selected.size} model(s) selected</Badge>
          <div className="flex gap-2">
            <Button variant="primary" size="sm" onClick={bulkEnrich} disabled={busy}>Enrich selected</Button>
            <Button variant="danger" size="sm" onClick={()=>setPendingDelete({kind:'bulk'})} disabled={busy}>Remove selected</Button>
            <Button variant="ghost" size="sm" onClick={clearSelection}>Clear</Button>
          </div>
        </div>
      )}

      {isAdmin && showAdd && (
        <Card className="flex flex-col sm:flex-row flex-wrap gap-2">
          <Select value={addProvider} onChange={e=>setAddProvider(e.target.value)} className="sm:w-48">
            <option value="">Provider</option>
            {providers.map((p:any)=> <option key={p.id} value={p.id}>{p.name}</option>)}
          </Select>
          <Input value={addModel} onChange={e=>setAddModel(e.target.value)} placeholder="model_id" className="flex-1 min-w-[160px] font-mono"/>
          <Button variant="primary" onClick={async()=>{
            if(!addProvider || !addModel){ toast.error('provider and model_id required'); return }
            try{
              await api.providerModels.add({provider_id:addProvider, model_id:addModel})
              setAddModel(''); await load()
              toast.success('Model added')
            }catch(e:any){ toast.error(e.message || String(e)) }
          }}>Add</Button>
        </Card>
      )}

      {/* Model cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3 md:gap-4">
        {loading && Array.from({ length: 6 }).map((_, i)=>(
          <Skeleton key={i} className="h-44" />
        ))}
        {!loading && list.map((m:any)=> {
          const levels = parseLevels(m.reasoning_levels)
          const limits = parseLimits(m.reasoning_output_limits)
          const isEdit = editId===m.id
          const isSelected = selected.has(m.id)
          const fullId = m.provider_name ? `${m.provider_name}/${m.model_id}` : m.model_id
          return (
            <Card key={m.id} pad={false} className={`p-4 flex flex-col transition-colors hover:border-muted/60 ${isSelected && !isEdit ? 'border-amber/40 bg-amber/5' : ''}`}>
              {/* Head */}
              <div className="flex items-start gap-2">
                {isAdmin && (
                  <input type="checkbox" checked={isSelected} onChange={()=>toggleSelect(m.id)} aria-label={`Select ${fullId}`} className="mt-1 accent-accent rounded shrink-0"/>
                )}
                <div className="min-w-0 flex-1">
                  <div className="font-medium truncate" title={m.display_name}>{m.display_name || fullId}</div>
                  <div className="mt-0.5 flex items-center gap-0.5 text-xs text-muted font-mono min-w-0" title={fullId}>
                    {m.provider_name ? <span className="shrink-0 text-muted/70">{m.provider_name}/</span> : null}
                    <span className="truncate">{m.model_id}</span>
                    <CopyButton value={fullId}/>
                  </div>
                </div>
                <Badge tone={sourceTone(m.source)}>{m.source}</Badge>
              </div>

              {/* Capability chips: reasoning (levels capped so rows stay aligned) + static capabilities from the API. */}
              {!isEdit && (() => {
                let mods: any = {}
                try { mods = JSON.parse(m.modalities || '{}') } catch { /* modalities is best-effort */ }
                const inputMods: string[] = Array.isArray(mods.input) ? mods.input : []
                const shown = levels.slice(0, 4)
                const hidden = levels.length - shown.length
                return (
                  <div className="mt-2.5 flex flex-wrap gap-1">
                    {m.reasoning ? (
                      <>
                        <Badge tone="info">reasoning: {m.reasoning_type||'effort'}</Badge>
                        {shown.map((lv:string)=> <Badge key={lv} tone="neutral">{lv}{limits[lv] ? ':'+fmtK(limits[lv]) : ''}</Badge>)}
                        {hidden > 0 && <span title={levels.join(', ')} className="inline-flex items-center text-xs font-medium px-2 py-0.5 rounded-full bg-stone/50 text-muted">+{hidden}</span>}
                      </>
                    ) : null}
                    {m.tool_call ? <Badge tone="neutral">tools</Badge> : null}
                    {inputMods.includes('image') ? <Badge tone="neutral">vision</Badge> : null}
                    {inputMods.includes('audio') ? <Badge tone="neutral">audio</Badge> : null}
                    {m.attachment ? <Badge tone="neutral">files</Badge> : null}
                  </div>
                )
              })()}

              {/* Inline editor (same fields and save normalization as before) */}
              {isEdit && (
                <div className="mt-3 space-y-2 rounded-lg border border-accent/30 bg-app/50 p-3">
                  <div className="grid grid-cols-2 gap-2">
                    <Field label="Context"><Input value={edit.context_window} inputMode="numeric" onChange={e=>setEdit({...edit, context_window: parseInt(e.target.value)||0})}/></Field>
                    <Field label="Max output"><Input value={edit.max_output} inputMode="numeric" onChange={e=>setEdit({...edit, max_output: parseInt(e.target.value)||0})}/></Field>
                    <Field label="Cost in"><Input value={edit.input_cost} inputMode="decimal" onChange={e=>setEdit({...edit, input_cost: parseFloat(e.target.value)||0})}/></Field>
                    <Field label="Cost out"><Input value={edit.output_cost} inputMode="decimal" onChange={e=>setEdit({...edit, output_cost: parseFloat(e.target.value)||0})}/></Field>
                  </div>
                  <label className="flex items-center gap-2 text-xs text-paper">
                    <input type="checkbox" checked={!!edit.reasoning} onChange={e=>setEdit({...edit, reasoning: e.target.checked})} className="accent-accent rounded"/> reasoning
                  </label>
                  {edit.reasoning && (
                    <div className="space-y-2">
                      <Select value={edit.reasoning_type||'effort'} onChange={e=>setEdit({...edit, reasoning_type:e.target.value})}>
                        <option value="effort">effort</option><option value="toggle">toggle</option><option value="none">none</option>
                      </Select>
                      <Input value={edit.reasoning_levels || '[]'} onChange={e=>setEdit({...edit, reasoning_levels: e.target.value})} className="font-mono text-xs"/>
                      <Input value={edit.reasoning_output_limits || '{}'} onChange={e=>setEdit({...edit, reasoning_output_limits: e.target.value})} className="font-mono text-xs"/>
                    </div>
                  )}
                </div>
              )}

              {/* Specs panel: 3 columns with nowrap values so prices/contexts never wrap mid-value. */}
              {!isEdit && (
                <div className="mt-3 rounded-lg border border-stone/70 bg-app/60 px-3 py-2.5 grid grid-cols-3 gap-2">
                  <div className="min-w-0">
                    <div className="text-[10px] uppercase tracking-wider text-muted">ctx</div>
                    <div className="font-mono text-xs mt-0.5 truncate" title={`${m.context_window ?? 0} tokens`}>{fmtK(m.context_window)}</div>
                  </div>
                  <div className="min-w-0">
                    <div className="text-[10px] uppercase tracking-wider text-muted">max out</div>
                    <div className="font-mono text-xs mt-0.5 truncate" title={`${m.max_output ?? 0} tokens`}>{fmtK(m.max_output)}</div>
                  </div>
                  <div className="min-w-0">
                    <div className="text-[10px] uppercase tracking-wider text-muted">cost /1M</div>
                    <div className="font-mono text-xs mt-0.5 whitespace-nowrap">{fmtCost(m.input_cost)} / {fmtCost(m.output_cost)}</div>
                  </div>
                </div>
              )}

              {/* Live traffic stats (populated when the model has served requests). */}
              {!isEdit && (
                <div className="mt-1.5 px-0.5 flex flex-wrap items-center gap-x-3 text-[11px] font-mono text-muted">
                  {m.avg_ttft_ms ? <span>ttft {fmtMs(m.avg_ttft_ms)}</span> : null}
                  {m.avg_tps ? <span>{m.avg_tps.toFixed(1)} tok/s</span> : null}
                  {m.request_count ? <span>{m.request_count} req</span> : null}
                  {!m.avg_ttft_ms && !m.avg_tps && !m.request_count ? <span>no traffic yet</span> : null}
                </div>
              )}

              {/* Actions (admin-only mutations) */}
              <div className="mt-auto pt-3 -mx-1 flex justify-end gap-1">
                {isAdmin && isEdit ? (
                  <>
                    <Button variant="ghost" size="sm" onClick={()=>setEditId(null)} title="Cancel edit">
                      <Icon name="x" size={14}/> Cancel
                    </Button>
                    <Button variant="primary" size="sm" onClick={async()=>{
                      let rl = edit.reasoning_levels; let rol = edit.reasoning_output_limits
                      if(rl && !rl.trim().startsWith('[')){ const arr = rl.split(',').map((s:string)=>s.trim()).filter(Boolean); rl = JSON.stringify(arr) }
                      if(rol && !rol.trim().startsWith('{')){ try{JSON.parse(rol)}catch{rol='{}'} }
                      try{
                        await api.providerModels.update(m.id, {...edit, reasoning_levels: rl, reasoning_output_limits: rol})
                        setEditId(null)
                        toast.success('Model updated')
                        load()
                      }catch(e:any){ toast.error(e.message || String(e)) }
                    }}>
                      <Icon name="check" size={14}/> Save
                    </Button>
                  </>
                ) : isAdmin ? (
                  <>
                    <Button variant="ghost" size="sm" onClick={async()=>{
                      try{ await api.providerModels.enrich(m.id); toast.success('Enriched'); load() }catch(e:any){ toast.error(e.message || String(e)) }
                    }} title="Enrich">
                      <Icon name="refresh" size={14}/> Enrich
                    </Button>
                    <Button variant="ghost" size="sm" onClick={()=>{ setEditId(m.id); setEdit({...m, reasoning_levels: m.reasoning_levels || '[]', reasoning_output_limits: m.reasoning_output_limits || '{}'})}} title="Edit">
                      <Icon name="pencil" size={14}/> Edit
                    </Button>
                    <Button variant="ghost" size="sm" onClick={()=>setPendingDelete({kind:'model', id:m.id, label:fullId})}
                      title={`Remove ${fullId} from discovery`} className="hover:text-red-400">
                      <Icon name="trash" size={14}/>
                    </Button>
                  </>
                ) : null}
              </div>
            </Card>
          )
        })}
        {!loading && !loadError && list.length===0 && (
          <div className="col-span-full">
            <EmptyState icon="box" title="No models." hint={isAdmin ? 'Discover from a provider or add a model manually above. Removed models sit in the recycling bin and stay out of discovery until restored.' : 'No models have been discovered yet — an admin can run discovery from the Providers page.'}/>
          </div>
        )}
      </div>

      {/* Recycling bin: removed models stay out of auto-discovery until restored. */}
      {(isAdmin || excluded.length > 0) && (
        <Card>
          <button type="button" onClick={()=>setBinOpen(o=>!o)} className="w-full flex items-center justify-between gap-3 text-left">
            <span className="flex items-center gap-2 font-semibold tracking-tight">
              <Icon name="trash" size={15}/> Recycling bin
              <Badge tone={excluded.length ? 'warn' : 'neutral'}>{excluded.length}</Badge>
            </span>
            <Icon name="chevronRight" size={14} className={`text-muted transition-transform ${binOpen ? 'rotate-90' : ''}`}/>
          </button>
          {binOpen && (
            excluded.length === 0 ? (
              <p className="mt-3 text-sm text-muted">Nothing here. Removed models land in the bin and are skipped by discovery until you restore them.</p>
            ) : (
              <div className="mt-3 divide-y divide-stone/70">
                {excluded.map((e:any)=>{
                  const snap = e.snapshot || {}
                  const fullId = e.provider_name ? `${e.provider_name}/${e.model_id}` : e.model_id
                  const removed = e.removed_at ? new Date(e.removed_at).toLocaleString() : ''
                  return (
                    <div key={e.id} className="py-2.5 flex items-center gap-3">
                      <div className="min-w-0 flex-1">
                        <div className="font-medium truncate" title={snap.display_name || fullId}>{snap.display_name || fullId}</div>
                        <div className="mt-0.5 flex items-center gap-2 text-xs text-muted font-mono min-w-0">
                          <span className="truncate" title={fullId}>{fullId}</span>
                          {removed && <span className="shrink-0 font-sans">removed {removed}</span>}
                        </div>
                      </div>
                      {snap.source && <Badge tone={sourceTone(snap.source)}>{snap.source}</Badge>}
                      {isAdmin && (
                        <Button variant="secondary" size="sm" disabled={restoring===e.id} onClick={async()=>{
                          setRestoring(e.id)
                          try{
                            await api.providerModels.restore(e.id)
                            toast.success(`Restored ${fullId}`)
                            await load()
                          }catch(err:any){ toast.error(err.message || String(err)) }
                          finally{ setRestoring(null) }
                        }}>
                          <Icon name="refresh" size={13}/> {restoring===e.id ? 'Restoring' : 'Restore'}
                        </Button>
                      )}
                    </div>
                  )
                })}
              </div>
            )
          )}
        </Card>
      )}

      {/* Aliases (writes are admin-only; everyone can read) */}
      <Card>
        <h3 className="font-semibold tracking-tight">Aliases</h3>
        {isAdmin && (
          <div className="mt-3 flex flex-col sm:flex-row flex-wrap gap-2">
            <Input value={newAlias} onChange={e=>setNewAlias(e.target.value)} placeholder="alias" className="sm:w-48"/>
            <Input value={newTarget} onChange={e=>setNewTarget(e.target.value)} placeholder="target model" className="flex-1 min-w-[200px]"/>
            <Button variant="primary" onClick={async()=>{
              try{
                await api.catalog.createAlias(newAlias, newTarget)
                setNewAlias(''); setNewTarget('')
                toast.success('Alias added')
                load()
              }catch(e:any){ toast.error(e.message || String(e)) }
            }}>Add</Button>
          </div>
        )}
        <div className={`${isAdmin ? 'mt-3' : 'mt-3'} flex flex-wrap gap-2 font-mono text-xs`}>
          {aliases.map((a:any)=> (
            <span key={a.alias} className="inline-flex items-center gap-1.5 border border-stone rounded-full pl-3 pr-1 py-1 bg-app/50">
              <b className="text-amber font-semibold">{a.alias}</b>
              <Icon name="chevronRight" size={11} className="text-muted"/>
              <span>{a.target}</span>
              {isAdmin && (
                <button onClick={()=>setPendingDelete({kind:'alias', alias:a.alias})} aria-label={`Delete alias ${a.alias}`}
                  className="ml-0.5 w-5 h-5 rounded-full flex items-center justify-center text-muted hover:text-red-400 hover:bg-stone/60 transition-colors">
                  <Icon name="x" size={12}/>
                </button>
              )}
            </span>
          ))}
          {aliases.length===0 && <span className="text-muted text-sm">No aliases.</span>}
        </div>
      </Card>

      <Confirm
        open={!!pendingDelete}
        onClose={()=>setPendingDelete(null)}
        onConfirm={confirmDelete}
        busy={busy}
        title={
          pendingDelete?.kind==='model' ? 'Remove model'
          : pendingDelete?.kind==='alias' ? 'Remove alias'
          : 'Remove selected models'
        }
        body={
          pendingDelete?.kind==='model' ? `Move "${pendingDelete.label}" to the recycling bin? It will not be auto-discovered again. Restore it from the bin to bring it back.`
          : pendingDelete?.kind==='alias' ? `Remove alias "${pendingDelete.alias}"? Requests resolving through it will stop mapping to its target.`
          : `Move ${selected.size} selected model(s) to the recycling bin? They will not be auto-discovered again. Restore them from the bin to bring them back.`
        }
        confirmLabel="Remove"
      />
    </div>
  )
}
