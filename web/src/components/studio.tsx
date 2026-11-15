"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { useTheme } from "next-themes";
import {
  ArrowDownToLine,
  ArrowRight,
  Box,
  Bone,
  Check,
  CircleAlert,
  Clock3,
  Film,
  ImagePlus,
  Play,
  Sparkles,
  Shuffle,
  ChevronRight,
  Layers3,
  LoaderCircle,
  Moon,
  Plus,
  RotateCcw,
  Search,
  SlidersHorizontal,
  Sun,
  Upload,
  Video,
  X,
} from "lucide-react";
import { Button, Input, Switch } from "./controls";

const Viewport = dynamic(() => import("./viewport"), { ssr: false });
const defaults = {
  target_vertices: 1000,
  faceted: true,
  rig: false,
  steps: 30,
  resolution: 256,
  seed: 42,
  remove_background: true,
  decode_chunk_size: 8000,
};
const motionDefaults = {
  prompt: "",
  duration: 5,
  steps: 50,
  seed: 42,
  root_motion: "in_place",
};
const videoDefaults = { prompt: "", seed: 42 };
const videoIdeas = [
  {
    label: "Product shot",
    prompt:
      "A small red toy car rolls slowly from left to right across a gray tabletop. Fixed side view, soft daylight, one continuous shot.",
  },
  {
    label: "Nature",
    prompt:
      "A single yellow flower sways gently in a light breeze. Close-up, fixed camera, soft morning light.",
  },
  {
    label: "Character",
    prompt:
      "A small clay robot raises one hand and waves. Full body, plain blue background, fixed camera.",
  },
];
function jobTitle(item: Job) {
  if (item.operation === "animation" && item.motion?.prompt)
    return item.motion.prompt;
  return item.operation === "video"
    ? item.settings.prompt?.split(/[.!?\n]/)[0] || "Untitled video"
    : item.filename.replace(/\.[^.]+$/, "");
}
const terminal = new Set(["completed", "failed", "cancelled"]);
type Asset = {
  id: string;
  filename: string;
  kind: string;
  thumbnail_url: string | null;
};
type Job = {
  id: string;
  asset_id: string | null;
  filename: string;
  kind: string;
  operation: "mesh" | "rig" | "animation" | "video";
  files: Record<string, string>;
  status: string;
  phase: string;
  progress: number;
  settings: Partial<typeof defaults> & { seed: number; prompt?: string };
  motion: typeof motionDefaults | null;
  parent_id: string | null;
  created_at: string;
  preview_url: string | null;
  original_url: string | null;
  thumbnail_url: string | null;
  error: string | null;
  stats: {
    vertices?: number;
    triangles?: number;
    reconstruction_reused?: boolean;
    rigging?: { bones: number };
    animation?: { contact_error_mean: number };
    simplification?: { warning?: string };
    video?: { width: number; height: number; duration: number };
  };
};

export default function Studio() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [assets, setAssets] = useState<Asset[]>([]);
  const [selected, setSelected] = useState("");
  const [settings, setSettings] = useState(defaults);
  const [motion, setMotion] = useState(motionDefaults);
  const [video, setVideo] = useState(videoDefaults);
  const [videoAsset, setVideoAsset] = useState<Asset | null>(null);
  const [mode, setMode] = useState("create");
  const [library, setLibrary] = useState(false);
  const [inspector, setInspector] = useState(true);
  const [queueOpen, setQueueOpen] = useState(false);
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [libraryFilter, setLibraryFilter] = useState("all");
  const [dragging, setDragging] = useState(false);
  const [view, setView] = useState("result");
  const input = useRef<HTMLInputElement>(null);
  const videoInput = useRef<HTMLInputElement>(null);
  const initialized = useRef(false);
  const modeSelections = useRef<Record<string, string>>({});
  const libraryDialog = useRef<HTMLDialogElement>(null);
  const dragDepth = useRef(0);
  const lock = useRef(false);
  const { resolvedTheme, setTheme } = useTheme();
  const job = jobs.find((item) => item.id === selected);
  const queue = jobs.filter((item) => !terminal.has(item.status));
  const preview =
    job?.status === "completed"
      ? job
      : jobs.find(
          (item) => item.id === job?.parent_id && item.status === "completed",
        );
  const ready = job?.status === "completed";
  const recent = jobs
    .filter(
      (item) =>
        item.status === "completed" &&
        (mode === "video"
          ? item.operation === "video"
          : mode === "animate"
            ? item.operation === "animation"
            : ["mesh", "rig"].includes(item.operation)),
    )
    .slice(0, 16);
  const groups = useMemo(() => {
    const grouped = new Map<string, Job[]>();
    for (const item of jobs) {
      const key = `${item.operation === "video" ? "video" : "mesh"}:${item.asset_id ?? item.id}`;
      grouped.set(key, [...(grouped.get(key) ?? []), item]);
    }
    return [...grouped.values()].filter(
      (items) =>
        (libraryFilter === "all" ||
          (libraryFilter === "video"
            ? items[0].operation === "video"
            : items[0].operation !== "video")) &&
        items.some((item) =>
          `${item.filename} ${item.settings.prompt ?? ""} ${item.motion?.prompt ?? ""}`
            .toLowerCase()
            .includes(query.toLowerCase()),
        ),
    );
  }, [jobs, query, libraryFilter]);

  useEffect(() => {
    const dialog = libraryDialog.current;
    if (!(library || queueOpen) || !dialog) return;
    const previousFocus = document.activeElement as HTMLElement | null;
    if (!dialog.open) dialog.showModal();
    return () => {
      if (dialog?.open) dialog.close();
      if (previousFocus?.isConnected) previousFocus.focus();
    };
  }, [library, queueOpen]);

  function closeLibrary() {
    setLibrary(false);
    setQueueOpen(false);
  }

  function changeMode(next: string) {
    modeSelections.current[mode] = selected;
    const saved = jobs.find((item) => item.id === modeSelections.current[next]);
    const compatible =
      next !== "video" && job?.operation !== "video" ? job : undefined;
    const candidate = saved ?? compatible;
    // Moving between workspaces should not replace an unfinished prompt.
    setSelected(candidate?.id ?? "");
    localStorage.setItem("studio.selection", candidate?.id ?? "");
    if (candidate && next !== "video")
      setSettings({ ...defaults, ...candidate.settings });
    setMode(next);
    setView("result");
    setInspector(true);
  }

  function newDraft() {
    select();
    setAssets([]);
    if (mode === "video") {
      setVideo(videoDefaults);
      setVideoAsset(null);
    } else if (mode === "animate") setMotion(motionDefaults);
    modeSelections.current[mode] = "";
    setInspector(true);
  }

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const response = await fetch("/api/jobs?limit=200", {
          signal: controller.signal,
        });
        if (!response.ok) throw new Error();
        const data = await response.json();
        if (controller.signal.aborted) return;
        setJobs(data.jobs);
        setConnected(true);
        if (!initialized.current) {
          const saved = localStorage.getItem("studio.selection");
          const initial =
            saved === ""
              ? undefined
              : (data.jobs.find((item: Job) => item.id === saved) ??
                data.jobs.find((item: Job) => item.status === "completed"));
          setSelected(initial?.id ?? "");
          setSettings({ ...defaults, ...initial?.settings });
          if (initial?.operation === "video") {
            setVideo({
              prompt: initial.settings.prompt ?? "",
              seed: initial.settings.seed,
            });
            setVideoAsset(
              initial.asset_id
                ? {
                    id: initial.asset_id,
                    filename: initial.filename,
                    kind: "image",
                    thumbnail_url: `/api/assets/${initial.asset_id}/thumbnail`,
                  }
                : null,
            );
            setMode("video");
          } else if (initial?.motion) {
            setMotion(initial.motion);
            setMode("animate");
          }
          initialized.current = true;
        }
      } catch {
        if (!controller.signal.aborted) setConnected(false);
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(poll, 2000);
      }
    }
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, []);

  function select(item?: Job) {
    setSelected(item?.id ?? "");
    localStorage.setItem("studio.selection", item?.id ?? "");
    setSettings({ ...defaults, ...item?.settings });
    if (item?.operation === "video") {
      setVideo({
        prompt: item.settings.prompt ?? "",
        seed: item.settings.seed,
      });
      setVideoAsset(
        item.asset_id
          ? {
              id: item.asset_id,
              filename: item.filename,
              kind: "image",
              thumbnail_url: `/api/assets/${item.asset_id}/thumbnail`,
            }
          : null,
      );
      setMode("video");
    } else if (item?.motion) {
      setMotion(item.motion);
      setMode("animate");
    } else if (item && mode === "video") {
      setMode("create");
    }
    setView("result");
    if (item) setAssets([]);
  }

  async function request(path: string, body?: object) {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    setError("");
    try {
      const response = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: body ? JSON.stringify(body) : undefined,
      });
      const data = await response.json();
      if (!response.ok)
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Check the settings and try again.",
        );
      setJobs((current) => [
        ...data.jobs,
        ...current.filter(
          (item) => !data.jobs.some((next: Job) => next.id === item.id),
        ),
      ]);
      select(data.jobs[0]);
      if (path === "/api/jobs") setAssets([]);
    } catch (failure) {
      setError(
        failure instanceof Error ? failure.message : "The request failed.",
      );
    } finally {
      lock.current = false;
      setBusy(false);
    }
  }

  async function upload(files: File[]) {
    if (lock.current || !files.length) return;
    if (
      mode === "video" &&
      files.some((file) => !/\.(png|jpe?g|webp)$/i.test(file.name))
    ) {
      setError("Choose a PNG, JPG, or WebP image for video.");
      return;
    }
    lock.current = true;
    setBusy(true);
    setError("");
    const uploaded: Asset[] = [];
    try {
      for (const file of files.slice(
        0,
        mode === "video" ? 1 : 16 - assets.length,
      )) {
        const body = new FormData();
        body.append("file", file);
        const response = await fetch("/api/assets", { method: "POST", body });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail);
        uploaded.push(data);
      }
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Upload failed.");
    } finally {
      if (uploaded.length) {
        if (mode === "video") {
          setVideoAsset(uploaded[0]);
          select();
        } else {
          setAssets((current) => [...current, ...uploaded]);
          select();
          setMode("create");
        }
        setInspector(true);
      }
      lock.current = false;
      setBusy(false);
    }
  }

  const previewUrl =
    view === "original"
      ? preview?.original_url
      : view === "rig"
        ? preview?.files["rig-glb"]
        : preview?.preview_url;
  return (
    <main
      className="studio stage"
      data-mode={mode}
      data-inspector={inspector}
      data-recent={recent.length > 0}
      data-dragging={dragging}
      onDragOver={(event) => event.preventDefault()}
      onDragEnter={(event) => {
        event.preventDefault();
        if (event.dataTransfer.types.includes("Files")) {
          dragDepth.current += 1;
          setDragging(true);
        }
      }}
      onDragLeave={(event) => {
        event.preventDefault();
        dragDepth.current = Math.max(0, dragDepth.current - 1);
        if (!dragDepth.current) setDragging(false);
      }}
      onDrop={(event) => {
        event.preventDefault();
        dragDepth.current = 0;
        setDragging(false);
        void upload(Array.from(event.dataTransfer.files));
      }}
    >
      <header className="app-header surface">
        <a className="brand" href="/" aria-label="Low Poly Studio">
          <span className="brand-mark">
            <Box size={20} />
          </span>
          <span>
            lowpoly<span className="brand-sub">STUDIO</span>
          </span>
        </a>
        <nav className="segment mode-switch" aria-label="Workspace mode">
          {[
            { value: "create", label: "Create", icon: Box },
            { value: "animate", label: "Animate", icon: Film },
            { value: "video", label: "Video", icon: Video },
          ].map(({ value, label, icon: Icon }) => (
            <Button
              key={value}
              variant="ghost"
              aria-pressed={mode === value}
              onClick={() => changeMode(value)}
            >
              <Icon />
              {label}
            </Button>
          ))}
        </nav>
        <div className="header-actions">
          <Button
            variant="ghost"
            className="library-toggle"
            title="Open your library"
            aria-label="Toggle library"
            aria-expanded={library}
            onClick={() => {
              setLibrary(!library);
              setQueueOpen(false);
            }}
          >
            <Layers3 />
            <span>Library</span>
          </Button>
          <Button
            variant="ghost"
            className="queue-toggle"
            title="Generation queue"
            aria-label="Open queue"
            aria-expanded={queueOpen}
            onClick={() => {
              setQueueOpen(!queueOpen);
              setLibrary(false);
            }}
          >
            <Clock3 />
            {queue.length > 0 && <span>{queue.length}</span>}
          </Button>
          <Button
            variant="ghost"
            size="icon"
            title="Switch light / dark theme"
            aria-label="Toggle theme"
            onClick={() =>
              setTheme(resolvedTheme === "dark" ? "light" : "dark")
            }
          >
            {resolvedTheme === "dark" ? <Sun /> : <Moon />}
          </Button>
          <Button
            variant="ghost"
            size="icon"
            title="Show or hide settings"
            aria-label="Toggle inspector"
            aria-expanded={inspector}
            onClick={() => setInspector(!inspector)}
          >
            <SlidersHorizontal />
          </Button>
        </div>
      </header>
      <section className="workspace" aria-label="Creative workspace">
        <div className="workspace-topbar">
          <div className="stage-heading">
            <div className="workspace-breadcrumb">
              <span>
                {mode === "video"
                  ? "Video studio"
                  : mode === "animate"
                    ? "Character animation"
                    : "3D studio"}
              </span>
              <ChevronRight size={13} />
              <span>{job ? "Library" : "New creation"}</span>
            </div>
            <h1 title={job ? jobTitle(job) : undefined}>
              {job
                ? jobTitle(job)
                : mode === "video"
                  ? "A new point of view."
                  : mode === "animate"
                    ? "Bring your character to life."
                    : "Make something with character."}
            </h1>
            <div className="output-meta">
              {preview?.stats.video ? (
                <>
                  <span>{preview.stats.video.duration.toFixed(1)} seconds</span>
                  <span>
                    {preview.stats.video.width} × {preview.stats.video.height}
                  </span>
                  <span>MP4</span>
                </>
              ) : preview ? (
                <>
                  <span>
                    {preview.stats.vertices?.toLocaleString()} vertices
                  </span>
                  <span>
                    {preview.stats.triangles?.toLocaleString()} triangles
                  </span>
                  {preview.stats.rigging && (
                    <span>{preview.stats.rigging.bones} bones</span>
                  )}
                </>
              ) : (
                <span>
                  {mode === "video"
                    ? "Your words. A little motion. Something new."
                    : mode === "animate"
                      ? "Start with a character from your library."
                      : "From a single image to a world of possibilities."}
                </span>
              )}
              {ready && (
                <span className="saved-badge">
                  <Check size={12} />
                  Saved
                </span>
              )}
            </div>
          </div>
          <div className="workspace-actions">
            <Button variant="outline" onClick={newDraft}>
              <Plus />
              {mode === "video"
                ? "New video"
                : mode === "animate"
                  ? "New animation"
                  : "New asset"}
            </Button>
            {ready && (
              <a
                className="download-button"
                href={
                  job.files.mp4 ??
                  job.files["animation-glb"] ??
                  job.files.glb ??
                  job.preview_url ??
                  "#"
                }
                download
              >
                <ArrowDownToLine size={16} />
                <span>
                  {job.operation === "video" ? "Download MP4" : "Download GLB"}
                </span>
              </a>
            )}
          </div>
        </div>
        <div className="preview-surface stage" data-empty={!preview}>
          <div className="preview-label">
            <span className="preview-dot" />
            {mode === "video" ? "VIDEO PREVIEW" : "INTERACTIVE PREVIEW"}
          </div>
          {mode === "video" ? (
            <div className="video-stage">
              {preview?.operation === "video" && previewUrl ? (
                <video
                  key={previewUrl}
                  src={previewUrl}
                  poster={preview.thumbnail_url ?? undefined}
                  controls
                  playsInline
                  preload="metadata"
                  aria-label="Generated video"
                />
              ) : (
                <div className="video-placeholder">
                  <div className="storyboard" aria-hidden="true">
                    <span>
                      <ImagePlus />
                    </span>
                    <span>
                      <Play fill="currentColor" />
                    </span>
                    <span>
                      <Film />
                    </span>
                  </div>
                  <span className="eyebrow">SMALL SCENES. BIG IDEAS.</span>
                  <h2>What happens next?</h2>
                  <p>
                    Describe a moment, or start with an image.
                    <br />
                    Bring it to life in a five-second video.
                  </p>
                  <div className="empty-specs">
                    <span>
                      <Clock3 size={13} />5 seconds
                    </span>
                    <span>
                      <Video size={13} />
                      320 × 192
                    </span>
                    <span>
                      <Check size={13} />
                      Saved locally
                    </span>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <Viewport
              url={preview?.operation === "video" ? null : (previewUrl ?? null)}
            />
          )}
          {mode !== "video" && !preview && !job && (
            <div className="empty-stage">
              <span className="empty-cube">
                <Box strokeWidth={1} />
              </span>
              <span className="eyebrow">
                {mode === "animate"
                  ? "A LITTLE PERSONALITY"
                  : "IT STARTS WITH AN IDEA"}
              </span>
              <h2>
                {mode === "animate"
                  ? "Give your character a story."
                  : "Your next creation starts here."}
              </h2>
              <p>
                {mode === "animate"
                  ? "Choose a saved character, then describe how it moves."
                  : "Drop in a reference image or bring your own 3D mesh."}
              </p>
              <Button
                onClick={() => {
                  if (mode === "animate") {
                    setLibraryFilter("mesh");
                    setLibrary(true);
                  } else input.current?.click();
                }}
                disabled={busy}
              >
                {mode === "animate" ? <Layers3 /> : <Upload />}
                {mode === "animate" ? "Choose a character" : "Import asset"}
              </Button>
              <span className="empty-formats">
                {mode === "animate"
                  ? "Rigging is included when needed"
                  : "PNG, JPG, WebP, GLB, OBJ · Up to 50 MB"}
              </span>
            </div>
          )}
          {job && !terminal.has(job.status) && (
            <div className="job-status surface" role="status">
              <LoaderCircle className="animate-spin" />
              <div>
                <strong>{job.phase}</strong>
                <progress value={job.progress} max={100} />
              </div>
              <Button
                variant="ghost"
                size="icon"
                aria-label="Cancel job"
                disabled={busy || job.status === "cancelling"}
                onClick={() => void request(`/api/jobs/${job.id}/cancel`)}
              >
                <X />
              </Button>
            </div>
          )}
          {job && ["failed", "cancelled"].includes(job.status) && (
            <div className="job-status surface">
              <CircleAlert />
              <div>
                <strong>
                  {job.status === "cancelled"
                    ? "Cancelled"
                    : "Couldn’t finish this version"}
                </strong>
                {job.error && <p>{job.error}</p>}
              </div>
              <Button
                variant="outline"
                disabled={busy}
                onClick={() => void request(`/api/jobs/${job.id}/retry`)}
              >
                <RotateCcw />
                Retry
              </Button>
            </div>
          )}

          <div className="preview-caption">
            {mode === "video"
              ? preview
                ? "Made in your local studio"
                : "5-second clips · Text or image to video"
              : "Drag to orbit · Scroll to zoom"}
          </div>
        </div>
        <section
          className="recent-section"
          aria-label={
            mode === "video"
              ? "Video clips"
              : mode === "animate"
                ? "Animation clips"
                : "Mesh versions"
          }
        >
          <div className="recent-heading">
            <h2>
              {mode === "video"
                ? "Recent videos"
                : mode === "animate"
                  ? "Recent animations"
                  : "Recent creations"}
              <span>{recent.length}</span>
            </h2>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                setLibraryFilter(mode === "video" ? "video" : "mesh");
                setLibrary(true);
              }}
            >
              View library
              <ArrowRight size={14} />
            </Button>
          </div>
          <div className="recent-list">
            {recent.map((item) => (
              <button
                className="recent-card"
                key={item.id}
                title={jobTitle(item)}
                aria-label={`Open ${jobTitle(item)}`}
                aria-pressed={item.id === selected}
                onClick={() => select(item)}
              >
                <div className="recent-image">
                  {item.thumbnail_url ? (
                    <img src={item.thumbnail_url} alt="" loading="lazy" />
                  ) : (
                    <Box />
                  )}
                  {["video", "animation"].includes(item.operation) && (
                    <span>
                      <Play size={11} fill="currentColor" />
                    </span>
                  )}
                </div>
                <div>
                  <strong>{jobTitle(item)}</strong>
                  <small>
                    {item.operation === "video"
                      ? `${item.stats.video?.duration.toFixed(1)} s · Video`
                      : item.operation === "animation"
                        ? `${item.motion?.duration} s · Animation`
                        : `${item.stats.vertices?.toLocaleString()} vertices`}
                  </small>
                </div>
                {selected === item.id && (
                  <Check className="recent-check" size={14} />
                )}
              </button>
            ))}
            {!recent.length && (
              <div className="recent-empty">
                <Layers3 size={18} />
                <span>
                  Your saved {mode === "video" ? "videos" : "creations"} will
                  appear here.
                </span>
              </div>
            )}
          </div>
        </section>
      </section>
      <input
        ref={videoInput}
        type="file"
        hidden
        accept=".png,.jpg,.jpeg,.webp"
        onChange={(event) => {
          void upload(Array.from(event.target.files ?? []));
          event.target.value = "";
        }}
      />
      <input
        ref={input}
        type="file"
        hidden
        multiple
        accept=".png,.jpg,.jpeg,.webp,.glb,.obj"
        onChange={(event) => {
          void upload(Array.from(event.target.files ?? []));
          event.target.value = "";
        }}
      />
      {(library || queueOpen) && (
        <dialog
          ref={libraryDialog}
          className="library surface"
          onCancel={closeLibrary}
          onClick={(event) => {
            if (event.target === event.currentTarget) {
              const bounds = event.currentTarget.getBoundingClientRect();
              if (
                event.clientX < bounds.left ||
                event.clientX > bounds.right ||
                event.clientY < bounds.top ||
                event.clientY > bounds.bottom
              )
                closeLibrary();
            }
          }}
          aria-label={library ? "Asset library" : "Job queue"}
        >
          <div className="panel-heading">
            <h2>{library ? "Your collection" : "In the queue"}</h2>
            <Button
              variant="ghost"
              size="icon"
              aria-label="Close library or queue"
              onClick={() => {
                setLibrary(false);
                setQueueOpen(false);
              }}
            >
              <X />
            </Button>
          </div>
          {library ? (
            <>
              <div className="search">
                <Search />
                <Input
                  aria-label="Search assets"
                  placeholder="Search names or prompts…"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
              </div>
              <Button
                variant="outline"
                className="wide"
                onClick={() => {
                  select();
                  setMode("create");
                  setLibrary(false);
                  setInspector(true);
                  input.current?.click();
                }}
              >
                <Plus />
                Import asset
              </Button>
              <div
                className="library-filters"
                role="group"
                aria-label="Filter library"
              >
                {[
                  { value: "all", label: "All" },
                  { value: "mesh", label: "3D & animation" },
                  { value: "video", label: "Videos" },
                ].map(({ value, label }) => (
                  <Button
                    key={value}
                    variant="ghost"
                    size="sm"
                    aria-pressed={libraryFilter === value}
                    onClick={() => setLibraryFilter(value)}
                  >
                    {label}
                  </Button>
                ))}
              </div>
              <div className="collection">
                {groups.map((items) => {
                  const item =
                    items.find((item) => item.status === "completed") ??
                    items[0];
                  return (
                    <button
                      className="asset-card"
                      key={item.id}
                      aria-pressed={items.some((item) => item.id === selected)}
                      onClick={() => {
                        select(item);
                        setLibrary(false);
                      }}
                    >
                      <div className="asset-image">
                        {item.thumbnail_url ? (
                          <img src={item.thumbnail_url} alt="" />
                        ) : item.operation === "video" ? (
                          <Video />
                        ) : (
                          <Box />
                        )}
                      </div>
                      <strong
                        title={
                          item.operation === "video"
                            ? jobTitle(item)
                            : item.filename
                        }
                      >
                        {item.operation === "video"
                          ? jobTitle(item)
                          : item.filename.replace(/\.[^.]+$/, "")}
                      </strong>
                      <span>
                        {item.operation === "video"
                          ? `${item.stats.video?.duration?.toFixed(1) ?? "5.1"} s · Video`
                          : `${items.length} ${items.length === 1 ? "version" : "versions"} · 3D asset`}
                      </span>
                      {item.status !== "completed" && (
                        <span className="asset-status">{item.phase}</span>
                      )}
                    </button>
                  );
                })}
                {!groups.length && (
                  <p className="library-empty">
                    {query
                      ? "No matches. Try another name or prompt."
                      : "Nothing here yet. Your saved creations will appear here."}
                  </p>
                )}
              </div>
            </>
          ) : (
            <div className="queue-list">
              {queue.length === 0 && <p className="muted">All caught up.</p>}
              {queue.map((item) => (
                <article key={item.id}>
                  <button onClick={() => select(item)}>
                    <strong>{item.filename}</strong>
                    <small>{item.phase}</small>
                  </button>
                  <progress max={100} value={item.progress} />
                  <Button
                    variant="ghost"
                    size="icon"
                    aria-label={`Cancel ${item.filename}`}
                    disabled={busy || item.status === "cancelling"}
                    onClick={() => void request(`/api/jobs/${item.id}/cancel`)}
                  >
                    <X />
                  </Button>
                </article>
              ))}
            </div>
          )}
        </dialog>
      )}
      {inspector && (
        <aside className="inspector surface" aria-label="Workspace inspector">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">
                {mode === "video"
                  ? "CREATE A VIDEO"
                  : mode === "animate"
                    ? "CREATE AN ANIMATION"
                    : "CREATE A 3D ASSET"}
              </span>
              <h2>
                {mode === "video"
                  ? "Make your next scene"
                  : mode === "animate"
                    ? "Set it in motion"
                    : "Shape your idea"}
              </h2>
            </div>
            <Button
              variant="ghost"
              size="icon"
              aria-label="Close inspector"
              onClick={() => setInspector(false)}
            >
              <X />
            </Button>
          </div>
          <form
            onKeyDown={(event) => {
              if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
                event.preventDefault();
                event.currentTarget.requestSubmit();
              }
            }}
            onSubmit={(event) => {
              event.preventDefault();
              if (
                busy ||
                !connected ||
                (mode === "video" && video.prompt.trim().length < 3)
              )
                return;
              if (mode === "video")
                void request("/api/videos", {
                  asset_id: videoAsset?.id ?? null,
                  settings: { prompt: video.prompt, seed: video.seed },
                });
              else if (mode === "animate" && ready && job.operation !== "video")
                void request(`/api/jobs/${job.id}/animations`, motion);
              else if (assets.length)
                void request("/api/jobs", {
                  asset_ids: assets.map((item) => item.id),
                  settings,
                });
              else if (ready)
                void request(`/api/jobs/${job.id}/variants`, {
                  target_vertices: settings.target_vertices,
                  faceted: settings.faceted,
                  rig: settings.rig,
                });
            }}
          >
            {mode === "video" ? (
              <>
                <div className="form-section-label">
                  <span>01</span>Starting point<small>Optional</small>
                </div>
                {videoAsset ? (
                  <div className="starting-image">
                    {videoAsset.thumbnail_url && (
                      <img
                        src={videoAsset.thumbnail_url}
                        alt="Starting frame"
                      />
                    )}
                    <div>
                      <span>{videoAsset.filename}</span>
                      <Button
                        variant="ghost"
                        size="icon"
                        aria-label="Remove starting image"
                        onClick={() => setVideoAsset(null)}
                      >
                        <X />
                      </Button>
                    </div>
                    <button
                      className="replace-image"
                      type="button"
                      onClick={() => videoInput.current?.click()}
                    >
                      Replace image
                    </button>
                  </div>
                ) : (
                  <button
                    type="button"
                    className="image-dropzone"
                    disabled={busy}
                    onClick={() => videoInput.current?.click()}
                  >
                    <span>
                      <ImagePlus size={22} />
                    </span>
                    <strong>Add a starting image</strong>
                    <small>Drop here or click to browse</small>
                    <span className="file-types">PNG · JPG · WebP</span>
                  </button>
                )}
                {videoAsset && (
                  <p className="field-help">
                    Your image sets the first frame and is cropped to fit.
                  </p>
                )}
                <div className="form-section-label">
                  <span>02</span>The scene
                </div>
                <label className="field prompt-field" htmlFor="video-prompt">
                  <span>
                    Video prompt
                    <span className="prompt-count" aria-hidden="true">
                      {video.prompt.length}/600
                    </span>
                  </span>
                  <textarea
                    id="video-prompt"
                    required
                    minLength={3}
                    maxLength={600}
                    rows={5}
                    placeholder="A red toy car rolls across a tabletop. Fixed camera, soft afternoon light…"
                    value={video.prompt}
                    onChange={(event) =>
                      setVideo({ ...video, prompt: event.target.value })
                    }
                  />
                </label>
                <p className="field-help">
                  One subject, a clear movement, and a camera direction.
                </p>
                <div
                  className="prompt-ideas video-ideas"
                  aria-label="Video prompt ideas"
                >
                  {videoIdeas.map((idea) => (
                    <Button
                      key={idea.label}
                      size="xs"
                      variant="ghost"
                      onClick={() =>
                        setVideo({ ...video, prompt: idea.prompt })
                      }
                    >
                      <Sparkles size={12} />
                      {idea.label}
                    </Button>
                  ))}
                </div>
                <div className="video-specs">
                  <div>
                    <span>Duration</span>
                    <strong>
                      5.1 <small>sec</small>
                    </strong>
                  </div>
                  <div>
                    <span>Resolution</span>
                    <strong>320 × 192</strong>
                  </div>
                  <div>
                    <span>Frame rate</span>
                    <strong>
                      16 <small>fps</small>
                    </strong>
                  </div>
                </div>
                <details className="advanced">
                  <summary>Advanced settings</summary>
                  <label className="field">
                    <span>Seed</span>
                    <div className="seed-input">
                      <Input
                        type="number"
                        min={0}
                        max={4294967295}
                        value={video.seed}
                        onChange={(event) =>
                          setVideo({
                            ...video,
                            seed: Number(event.target.value),
                          })
                        }
                      />
                      <Button
                        variant="outline"
                        size="icon"
                        title="Randomize seed"
                        aria-label="Randomize seed"
                        onClick={() =>
                          setVideo({
                            ...video,
                            seed: crypto.getRandomValues(new Uint32Array(1))[0],
                          })
                        }
                      >
                        <Shuffle />
                      </Button>
                    </div>
                  </label>
                  <p className="field-help">
                    Reuse a seed and prompt for a similar result.
                  </p>
                </details>
                <div className="generate-footer">
                  <Button
                    type="submit"
                    className="primary-action"
                    disabled={
                      busy || !connected || video.prompt.trim().length < 3
                    }
                  >
                    {busy ? (
                      <LoaderCircle className="animate-spin" />
                    ) : (
                      <Sparkles />
                    )}
                    Generate video
                    <ArrowRight />
                  </Button>
                  <p>
                    Saved to your library <span>Ctrl / ⌘ + Enter</span>
                  </p>
                </div>
              </>
            ) : mode === "create" ? (
              <>
                {assets.length > 0 ? (
                  <div className="uploads">
                    {assets.map((asset) => (
                      <div key={asset.id}>
                        {asset.thumbnail_url && (
                          <img src={asset.thumbnail_url} alt="" />
                        )}
                        <span>{asset.filename}</span>
                        <Button
                          variant="ghost"
                          size="icon"
                          aria-label={`Remove ${asset.filename}`}
                          onClick={() =>
                            setAssets((current) =>
                              current.filter((item) => item.id !== asset.id),
                            )
                          }
                        >
                          <X />
                        </Button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <Button
                    variant="outline"
                    className="wide import-button"
                    disabled={busy}
                    onClick={() => input.current?.click()}
                  >
                    <Upload />
                    Import image or mesh
                    <Plus />
                  </Button>
                )}
                <label className="field">
                  <span>
                    Vertex budget
                    <output>
                      {settings.target_vertices === 0
                        ? "Original"
                        : settings.target_vertices.toLocaleString()}
                    </output>
                  </span>
                  <input
                    type="range"
                    min={100}
                    max={5000}
                    step={100}
                    aria-label="Vertex budget"
                    value={settings.target_vertices || 5000}
                    onChange={(event) =>
                      setSettings({
                        ...settings,
                        target_vertices: Number(event.target.value),
                      })
                    }
                  />
                </label>
                <div className="segment budgets">
                  {[500, 1000, 2000, 0].map((value) => (
                    <Button
                      variant="ghost"
                      key={value}
                      aria-pressed={settings.target_vertices === value}
                      onClick={() =>
                        setSettings({ ...settings, target_vertices: value })
                      }
                    >
                      {value || "Original"}
                    </Button>
                  ))}
                </div>
                <label className="switch-row">
                  Faceted surface
                  <Switch
                    checked={settings.faceted}
                    onCheckedChange={(faceted) =>
                      setSettings({ ...settings, faceted })
                    }
                  />
                </label>
                <label className="switch-row">
                  Rig as a character
                  <Switch
                    checked={settings.rig}
                    onCheckedChange={(rig) => setSettings({ ...settings, rig })}
                  />
                </label>
                <details className="advanced">
                  <summary>Generation settings</summary>
                  <label className="field">
                    <span>Exact vertex budget</span>
                    <Input
                      type="number"
                      min={0}
                      max={200000}
                      value={settings.target_vertices}
                      onChange={(e) =>
                        setSettings({
                          ...settings,
                          target_vertices: Number(e.target.value),
                        })
                      }
                    />
                  </label>
                  <label className="field">
                    <span>Shape steps</span>
                    <Input
                      type="number"
                      min={1}
                      max={50}
                      value={settings.steps}
                      disabled={ready && !assets.length}
                      onChange={(e) =>
                        setSettings({
                          ...settings,
                          steps: Number(e.target.value),
                        })
                      }
                    />
                  </label>
                  <label className="field">
                    <span>Resolution</span>
                    <select
                      value={settings.resolution}
                      disabled={ready && !assets.length}
                      onChange={(e) =>
                        setSettings({
                          ...settings,
                          resolution: Number(e.target.value),
                        })
                      }
                    >
                      {[128, 256, 384].map((value) => (
                        <option key={value}>{value}</option>
                      ))}
                    </select>
                  </label>
                  <label className="field">
                    <span>Seed</span>
                    <Input
                      type="number"
                      min={0}
                      max={4294967295}
                      value={settings.seed}
                      disabled={ready && !assets.length}
                      onChange={(e) =>
                        setSettings({
                          ...settings,
                          seed: Number(e.target.value),
                        })
                      }
                    />
                  </label>
                  <label className="switch-row">
                    Remove background
                    <Switch
                      checked={settings.remove_background}
                      disabled={ready && !assets.length}
                      onCheckedChange={(remove_background) =>
                        setSettings({ ...settings, remove_background })
                      }
                    />
                  </label>
                </details>
                <div className="generate-footer">
                  <Button
                    type="submit"
                    className="primary-action"
                    disabled={busy || (!ready && !assets.length)}
                  >
                    {busy ? <LoaderCircle className="animate-spin" /> : <Box />}
                    {assets.length
                      ? `Create ${assets.length > 1 ? `${assets.length} assets` : "asset"}`
                      : "Save mesh version"}
                    <ArrowRight />
                  </Button>
                </div>
              </>
            ) : (
              <>
                <div className="character-slot">
                  <span className="slot-icon">
                    <Bone />
                  </span>
                  <div>
                    <strong>
                      {preview?.filename.replace(/\.[^.]+$/, "") ??
                        "Choose a character"}
                    </strong>
                    <small>
                      {preview?.stats.rigging
                        ? "Rig ready"
                        : preview
                          ? "Rigging included"
                          : "Open your collection to get started"}
                    </small>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    aria-label="Choose character"
                    onClick={() => setLibrary(true)}
                  >
                    <Layers3 />
                  </Button>
                </div>
                <label className="field">
                  <span>Motion prompt</span>
                  <textarea
                    required
                    minLength={3}
                    maxLength={600}
                    rows={4}
                    placeholder="A person takes a small bow, then waves hello."
                    value={motion.prompt}
                    onChange={(event) =>
                      setMotion({ ...motion, prompt: event.target.value })
                    }
                  />
                </label>
                <div className="prompt-ideas">
                  {["Walk", "Jump", "Wave"].map((name, index) => (
                    <Button
                      key={name}
                      size="xs"
                      variant="ghost"
                      onClick={() =>
                        setMotion({
                          ...motion,
                          prompt: [
                            "A person walks forward at a relaxed, steady pace.",
                            "A person jumps straight up with both feet and lands on both feet.",
                            "A person stands in place and waves hello with their right hand.",
                          ][index],
                        })
                      }
                    >
                      {name}
                    </Button>
                  ))}
                </div>
                <label className="field">
                  <span>
                    Duration<output>{motion.duration} seconds</output>
                  </span>
                  <input
                    type="range"
                    min={2}
                    max={10}
                    aria-label="Motion duration"
                    value={motion.duration}
                    onChange={(event) =>
                      setMotion({
                        ...motion,
                        duration: Number(event.target.value),
                      })
                    }
                  />
                </label>
                <div className="field">
                  <span>Quality</span>
                  <div className="segment">
                    {[50, 100].map((steps) => (
                      <Button
                        key={steps}
                        variant="ghost"
                        aria-pressed={motion.steps === steps}
                        onClick={() => setMotion({ ...motion, steps })}
                      >
                        {steps === 50 ? "Fast" : "Refined"}
                        <small>{steps} steps</small>
                      </Button>
                    ))}
                  </div>
                </div>
                <div className="field">
                  <span>Root motion</span>
                  <div className="segment">
                    {[
                      { value: "in_place", label: "In place" },
                      { value: "travel", label: "Travel" },
                    ].map(({ value, label }) => (
                      <Button
                        key={value}
                        variant="ghost"
                        aria-pressed={motion.root_motion === value}
                        onClick={() =>
                          setMotion({ ...motion, root_motion: value })
                        }
                      >
                        {label}
                      </Button>
                    ))}
                  </div>
                </div>
                <details className="advanced">
                  <summary>Reproducibility</summary>
                  <label className="field">
                    <span>Seed</span>
                    <Input
                      type="number"
                      min={0}
                      max={4294967295}
                      value={motion.seed}
                      onChange={(event) =>
                        setMotion({
                          ...motion,
                          seed: Number(event.target.value),
                        })
                      }
                    />
                  </label>
                </details>
                <div className="generate-footer">
                  <Button
                    type="submit"
                    className="primary-action"
                    disabled={
                      busy ||
                      !ready ||
                      job?.operation === "video" ||
                      motion.prompt.trim().length < 3
                    }
                  >
                    {busy ? (
                      <LoaderCircle className="animate-spin" />
                    ) : (
                      <Film />
                    )}
                    Generate animation
                    <ArrowRight />
                  </Button>
                </div>
              </>
            )}
          </form>
          {ready && (
            <div
              className="result-tools"
              data-video={job.operation === "video"}
            >
              <div className="seam-label">
                <Check />
                Saved to your library
              </div>
              {job.operation !== "video" && (
                <div className="segment">
                  <Button
                    variant="ghost"
                    aria-pressed={view === "result"}
                    onClick={() => setView("result")}
                  >
                    {job.operation === "animation" ? "Motion" : "Mesh"}
                  </Button>
                  {job.files["rig-glb"] && (
                    <Button
                      variant="ghost"
                      aria-pressed={view === "rig"}
                      onClick={() => setView("rig")}
                    >
                      Bind pose
                    </Button>
                  )}
                  {job.original_url && (
                    <Button
                      variant="ghost"
                      aria-pressed={view === "original"}
                      onClick={() => setView("original")}
                    >
                      Original
                    </Button>
                  )}
                </div>
              )}
              {mode === "create" && (
                <Button
                  variant="outline"
                  className="wide"
                  onClick={() => {
                    setMode("animate");
                    setMotion(motionDefaults);
                  }}
                >
                  <Film />
                  Animate this character
                  <ArrowRight />
                </Button>
              )}
              <details className="exports">
                <summary>
                  <ArrowDownToLine />
                  Download files
                </summary>
                <div>
                  {Object.entries(job.files).map(([name, url]) => (
                    <a key={name} href={url} download>
                      {name.replaceAll("-", " · ").toUpperCase()}
                      <ArrowDownToLine size={13} />
                    </a>
                  ))}
                </div>
              </details>
            </div>
          )}
          {preview?.stats.simplification?.warning && (
            <p className="notice">{preview.stats.simplification.warning}</p>
          )}
          {preview?.stats.animation &&
            preview.stats.animation.contact_error_mean > 0.05 && (
              <p className="notice">
                Some foot contacts exceed this rig’s reach. Check the legs
                during playback.
              </p>
            )}
        </aside>
      )}
      {error && (
        <div className="toast surface" role="alert">
          <CircleAlert />
          <span>{error}</span>
          <Button
            variant="ghost"
            size="icon"
            aria-label="Dismiss error"
            onClick={() => setError("")}
          >
            <X />
          </Button>
        </div>
      )}
      <footer className="stage-footer">
        <span className={connected ? "connection online" : "connection"} />
        <span>
          {connected
            ? queue.length
              ? `${queue.length} ${queue.length === 1 ? "job" : "jobs"} in queue`
              : "All systems ready"
            : "Reconnecting…"}
        </span>
        <span className="footer-local">
          Local workspace<span>Files stay on your machine</span>
        </span>
      </footer>
      {dragging && (
        <div className="drop-overlay">
          <Upload size={32} />
          <strong>
            {mode === "video"
              ? "Drop your starting image"
              : "Drop images or meshes"}
          </strong>
          <span>
            {mode === "video"
              ? "PNG, JPG, or WebP"
              : "PNG, JPG, WebP, GLB, or OBJ"}
          </span>
        </div>
      )}
    </main>
  );
}
