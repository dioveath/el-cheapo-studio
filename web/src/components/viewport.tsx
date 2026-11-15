"use client";

import {
  Component,
  Suspense,
  useEffect,
  useCallback,
  useMemo,
  useRef,
  useState,
  type ReactNode,
  type ComponentRef,
} from "react";
import { Canvas } from "@react-three/fiber";
import { useTheme } from "next-themes";
import {
  ContactShadows,
  GizmoHelper,
  GizmoViewport,
  Grid,
  OrbitControls,
  useGLTF,
} from "@react-three/drei";
import {
  AnimationMixer,
  Box3,
  DoubleSide,
  Mesh,
  MeshStandardMaterial,
  SkeletonHelper,
  Group,
  Vector3,
} from "three";
import { clone } from "three/addons/utils/SkeletonUtils.js";
import {
  Bone,
  Expand,
  LoaderCircle,
  Pause,
  Play,
  RotateCcw,
  ScanLine,
} from "lucide-react";
import { Button } from "./controls";

class SceneBoundary extends Component<
  { children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? (
      <div className="viewport-failure">
        This browser could not display the preview. Your exports are available.
      </div>
    ) : (
      this.props.children
    );
  }
}

function Scene({ url }: { url: string }) {
  const { scene, animations } = useGLTF(url);
  const { resolvedTheme } = useTheme();
  const [wireframe, setWireframe] = useState(false);
  const [skeleton, setSkeleton] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [reset, setReset] = useState(0);
  const timeline = useRef<HTMLInputElement>(null);
  const elapsed = useRef<HTMLOutputElement>(null);
  const invalidate = useRef<() => void>(() => {});
  const controls = useRef<ComponentRef<typeof OrbitControls>>(null);
  const ground = useRef<Group>(null);
  const { object, material, bones, mixer, action } = useMemo(() => {
    const material = new MeshStandardMaterial({
      color: "#c0a28b",
      roughness: 0.86,
      metalness: 0,
      side: DoubleSide,
    });
    const object = clone(scene);
    object.traverse((child) => {
      if (child instanceof Mesh) child.material = material;
    });
    const mixer = new AnimationMixer(object);
    const action = animations[0]
      ? mixer.clipAction(animations[0]).play()
      : null;
    mixer.update(0);
    object.updateMatrixWorld(true);
    const bounds = new Box3().setFromObject(object, true);
    const size = bounds.getSize(new Vector3());
    const scale = 2 / Math.max(size.x, size.y, size.z, 0.001);
    object.scale.multiplyScalar(scale);
    object.position.sub(bounds.getCenter(new Vector3()).multiplyScalar(scale));
    object.position.y += (size.y * scale) / 2 - 1;
    object.updateMatrixWorld(true);
    const bones = new SkeletonHelper(object);
    for (const material of [bones.material].flat()) {
      material.depthTest = false;
      material.transparent = true;
      material.opacity = 0.9;
    }
    bones.renderOrder = 1;
    return { object, material, bones, mixer, action };
  }, [scene, animations]);
  const duration = animations[0]?.duration ?? 0;

  useEffect(() => {
    material.wireframe = wireframe;
    bones.visible = skeleton;
    invalidate.current();
  }, [material, bones, wireframe, skeleton]);
  const seek = useCallback(
    (time: number) => {
      const root = bones.bones[0];
      const before = root?.getWorldPosition(new Vector3());
      mixer.setTime(time);
      object.updateMatrixWorld(true);
      if (root && before && controls.current) {
        const after = root.getWorldPosition(new Vector3());
        const delta = after.clone().sub(before).setY(0);
        controls.current.object.position.add(delta);
        controls.current.target.add(delta);
        controls.current.update();
        if (ground.current) ground.current.position.set(after.x, 0, after.z);
      }
      if (timeline.current) timeline.current.value = String(action?.time ?? 0);
      if (elapsed.current)
        elapsed.current.value = `${(action?.time ?? 0).toFixed(2)} s`;
      invalidate.current();
    },
    [mixer, object, bones, action],
  );
  useEffect(() => {
    if (!playing || !action) return;
    let frame: number;
    let previous = performance.now();
    function tick(now: number) {
      seek(mixer.time + Math.min((now - previous) / 1000, 0.1) * speed);
      previous = now;
      frame = requestAnimationFrame(tick);
    }
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, mixer, action, speed, seek]);
  useEffect(
    () => () => {
      mixer.stopAllAction();
      mixer.uncacheRoot(object);
      material.dispose();
      bones.dispose();
    },
    [object, material, bones, mixer],
  );

  return (
    <>
      <Canvas
        key={reset}
        camera={{ position: [3, 2.1, 4], fov: 34 }}
        dpr={[1, 2]}
        shadows
        frameloop="demand"
        gl={{ antialias: true }}
        aria-label="Interactive 3D mesh preview"
        onCreated={(state) => {
          invalidate.current = state.invalidate;
          state.camera.layers.enable(1);
        }}
      >
        <ambientLight intensity={0.95} />
        <directionalLight
          position={[-3, 6, 5]}
          intensity={2.3}
          color="#fff1df"
        />
        <directionalLight
          position={[4, 2, -3]}
          intensity={1.5}
          color="#dfebff"
        />
        <primitive object={object} />
        <primitive object={bones} />
        <Grid
          args={[100, 100]}
          position={[0, -1.02, 0]}
          layers-mask={2}
          cellSize={1}
          sectionSize={10}
          cellColor={resolvedTheme === "dark" ? "#776655" : "#a99c8e"}
          sectionColor={resolvedTheme === "dark" ? "#c28f6e" : "#a45f43"}
          cellThickness={0.6}
          sectionThickness={1}
          fadeDistance={80}
          fadeStrength={1.5}
        />
        <group ref={ground}>
          <ContactShadows
            position={[0, -1.015, 0]}
            opacity={0.3}
            scale={8}
            blur={2.8}
            far={3}
            resolution={256}
            color="#362218"
          />
        </group>
        <OrbitControls
          ref={controls}
          makeDefault
          minDistance={0.3}
          maxDistance={20}
        />
        <GizmoHelper
          alignment="bottom-right"
          margin={[45, duration ? 130 : 45]}
        >
          <GizmoViewport
            axisColors={["#b96d54", "#92996c", "#7d97aa"]}
            labelColor="#fffaf4"
            hideNegativeAxes
          />
        </GizmoHelper>
      </Canvas>
      <div className="viewport-tools" aria-label="Preview controls">
        <Button
          variant="ghost"
          size="icon"
          aria-label="Toggle wireframe"
          title="Wireframe"
          aria-pressed={wireframe}
          onClick={() => setWireframe((current) => !current)}
        >
          <ScanLine />
        </Button>
        {bones.bones.length > 0 && (
          <Button
            variant="ghost"
            size="icon"
            aria-label="Toggle skeleton"
            title="Skeleton"
            aria-pressed={skeleton}
            onClick={() => setSkeleton((current) => !current)}
          >
            <Bone />
          </Button>
        )}
        <span />
        <Button
          variant="ghost"
          size="icon"
          aria-label="Reset view"
          title="Reset view"
          onClick={() => setReset((current) => current + 1)}
        >
          <RotateCcw />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          aria-label="Expand preview"
          title="Expand preview"
          onClick={(event) => {
            const stage = event.currentTarget.closest(".stage");
            if (document.fullscreenElement) void document.exitFullscreen();
            else void stage?.requestFullscreen().catch(() => {});
          }}
        >
          <Expand />
        </Button>
      </div>
      {duration > 0 && (
        <div className="motion-dock" aria-label="Animation controls">
          <Button
            className="motion-play"
            variant="ghost"
            size="icon"
            aria-label={playing ? "Pause animation" : "Play animation"}
            onClick={() => setPlaying((current) => !current)}
          >
            {playing ? <Pause size={17} /> : <Play size={17} />}
          </Button>
          <div className="motion-track">
            <div className="motion-caption">
              <strong>{animations[0].name}</strong>
              <span>Loop · {duration.toFixed(2)} s</span>
            </div>
            <input
              ref={timeline}
              type="range"
              min={0}
              max={duration}
              step={0.001}
              defaultValue={0}
              aria-label="Animation position"
              onChange={(event) => {
                setPlaying(false);
                seek(Number(event.target.value));
              }}
            />
          </div>
          <select
            aria-label="Playback speed"
            value={speed}
            onChange={(event) => setSpeed(Number(event.target.value))}
          >
            {[0.25, 0.5, 1, 1.5, 2].map((value) => (
              <option key={value} value={value}>
                {value}×
              </option>
            ))}
          </select>
          <output ref={elapsed} aria-label="Animation time">
            0.00 s
          </output>
        </div>
      )}
    </>
  );
}

export default function Viewport({ url }: { url: string | null }) {
  return (
    <div className="viewport">
      {url && (
        <SceneBoundary key={url}>
          <Suspense
            fallback={
              <div className="viewport-loading">
                <LoaderCircle
                  className="animate-spin"
                  aria-label="Loading mesh"
                />
              </div>
            }
          >
            <Scene url={url} />
          </Suspense>
        </SceneBoundary>
      )}
    </div>
  );
}
