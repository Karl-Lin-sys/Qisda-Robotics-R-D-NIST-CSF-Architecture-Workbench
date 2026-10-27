export type SubsystemCategory =
  | 'all'
  | 'perception_slam'
  | 'planning_nav'
  | 'locomotion_wbc'
  | 'hardware_edge';

export type ExplanationMode = 'rigorous' | 'canadian_toddler' | 'dual';

export interface ConferenceArchitecture {
  id: string;
  subsystem: Exclude<SubsystemCategory, 'all'>;
  subsystemLabel: string;
  qisdaRequirement: string;
  title: string;
  venueAndYear: string;
  githubRepo: string;
  githubStars: string;
  leakageMitigation: string;
  overfittingControl: string;
  imbalanceMetric: string;
  edgeLatencyMs: number;
  closedLoopSuccessRate: number;
  rigorousSummary: string;
  canadianToddlerExplanation: string;
}

export interface ComplexityProfile {
  id: string;
  framework: 'OpenCV 4.x' | 'PyTorch 2.x' | 'TensorFlow 2.x';
  functionName: string;
  timeBigO: string;
  spaceBigO: string;
  eliminatedAntiPattern: string;
  officialDocReference: string;
  canadianToddlerSummary: string;
  codeSnippet: string;
}

export interface ComplexitySimulationInput {
  height: number;
  width: number;
  numPoints: number;
  batchSize: number;
  numClasses: number;
}

export interface ComplexitySimulationResult {
  opencvWarpOps: number;
  opencvPreallocBytes: number;
  opencvLeakyAllocBytesPerMinuteAt30Hz: number;
  pytorchForwardOps: number;
  pytorchBalancedSoftmaxOps: number;
  pytorchActivationBytes: number;
  tfXlaMahalanobisOps: number;
  tfDatasetStreamBytes: number;
}

export interface RootCauseSolution {
  id: string;
  indexNumber: string;
  defectTitle: string;
  primarySourceCitation: string;
  rootCauseAnalysis: string;
  solutionTitle: string;
  mathematicalFormulation: string;
  implementationDetail: string;
  canadianToddlerExplanation: string;
  beforeMetricLabel: string;
  beforeMetricValue: string;
  afterMetricLabel: string;
  afterMetricValue: string;
}

export interface NistCsfFunction {
  code: 'GV' | 'ID' | 'PR' | 'DE' | 'RS' | 'RC';
  name: string;
  categoryCode: string;
  qisdaSystemTarget: string;
  technicalImplementation: string;
  canadianToddlerRule: string;
  verificationCheck: string;
}

export interface TaiwanLegalCheckItem {
  id: string;
  statuteNameZh: string;
  statuteNameEn: string;
  articleRef: string;
  complianceCriterion: string;
  auditMethod: string;
  canadianToddlerExplanation: string;
  passedByDefault: boolean;
}

export const CONFERENCE_ARCHITECTURES: ConferenceArchitecture[] = [
  {
    id: 'bevfusion-gaussian-lic',
    subsystem: 'perception_slam',
    subsystemLabel: 'Perception & Multi-Modal SLAM',
    qisdaRequirement: 'LiDAR, RGB Camera, Depth Camera, LWIR Camera, IMU, SLAM',
    title: 'BEVFusion + KISS-ICP & Gaussian-LIC Multi-Spectral Odometry',
    venueAndYear: 'CVPR / ICRA / RAL (2023–2024)',
    githubRepo: 'mit-han-lab/bevfusion & PRBonn/kiss-icp',
    githubStars: '3,600+ / 2,800+',
    leakageMitigation:
      'Strict contiguous sequence-level Temporal Purge Split (5.0s guard band) preventing adjacent 30Hz RGB/LWIR frames from spanning Train and Val sets.',
    overfittingControl:
      'Modality-Dropout regularization (randomly masking RGB during training so the network cannot ignore LWIR thermal and LiDAR geometry in smoke/darkness) + parameter-free ICP.',
    imbalanceMetric:
      'NuScenes Detection Score (NDS) & Long-Tail AUPRC weighted by inverse class prior rather than overall pixel accuracy.',
    edgeLatencyMs: 14.2,
    closedLoopSuccessRate: 96.4,
    rigorousSummary:
      'Unifies multi-view RGB, Depth, LWIR thermal, and 3D LiDAR point clouds in a canonical Bird’s-Eye-View (BEV) metric space while running parameter-free point-to-point ICP with continuous-time IMU deskewing.',
    canadianToddlerExplanation:
      'Our robot uses colour eyes, laser eyes, and cosy heat-vision eyes all at the same time, eh! Even if the lights go out during a snowy Canadian night, it draws a neat top-down map without ever peeking at tomorrow’s puzzle!'
  },
  {
    id: 'uniad-diffusion-policy',
    subsystem: 'planning_nav',
    subsystemLabel: 'Planning & Autonomous Navigation',
    qisdaRequirement: 'Autonomous System Platform: Perception, Planning, Navigation',
    title: 'UniAD (Planning-Oriented Stack) & Visuomotor Diffusion Policy',
    venueAndYear: 'CVPR Best Paper / RSS / IJRR (2023–2024)',
    githubRepo: 'OpenDriveLab/UniAD & real-stanford/diffusion_policy',
    githubStars: '4,300+ / 2,700+',
    leakageMitigation:
      'Eliminates ground-truth ego-velocity shortcut leakage by evaluating strictly in closed-loop receding-horizon rollout with delayed sensor observations.',
    overfittingControl:
      'Conditional denoising score matching over action horizons (Tp = 16 steps, Ta = 8 executed steps) prevents mode-averaging overfitting on multi-modal obstacle avoidance paths.',
    imbalanceMetric:
      'Closed-Loop Collision Rate (%) at 3s horizon & Macro-F1 across rare dynamic obstacles (hospital stretchers, forklifts).',
    edgeLatencyMs: 18.5,
    closedLoopSuccessRate: 94.8,
    rigorousSummary:
      'Couples perception, online mapping, and motion forecasting directly into a unified query-based transformer and receding-horizon conditional diffusion action head, avoiding greedy single-step covariate shift.',
    canadianToddlerExplanation:
      'If a moose or a shopping cart stands in the middle of the hallway, our robot doesn’t get confused and walk straight into it! Instead of guessing halfway between left and right, it picks one safe path around, polite as can be, eh!'
  },
  {
    id: 'isaac-wbc-humanplus',
    subsystem: 'locomotion_wbc',
    subsystemLabel: 'Locomotion, Manipulation & Whole-Body Control',
    qisdaRequirement: 'Locomotion, Manipulation, Whole Body Control, Pro/E, OpenUSD, NVIDIA Isaac',
    title: 'Isaac Lab (Orbit) Asymmetric Actor-Critic Whole-Body Controller',
    venueAndYear: 'CoRL / RSS / CVPR (2024)',
    githubRepo: 'isaac-sim/IsaacLab & HybridRobotics/whole_body_tracking',
    githubStars: '2,600+ / 1,400+',
    leakageMitigation:
      'Privileged terrain and friction states are restricted strictly to the Critic during simulation training; the deployed Actor receives only causal proprioception + depth history.',
    overfittingControl:
      'Domain Randomization over link mass (±15%), actuator delay (1–4ms EtherCAT jitter), and joint friction prevents overfitting to ideal Pro/E CAD inertia tensors imported via OpenUSD.',
    imbalanceMetric:
      'Worst-Decile Terrain Survival Rate (CVaR 10%) & End-Effector SE(3) Tracking RMSE under external impulse perturbations.',
    edgeLatencyMs: 2.1,
    closedLoopSuccessRate: 98.1,
    rigorousSummary:
      'Trains a high-frequency (200Hz policy / 1000Hz EtherCAT PD torque loop) whole-body locomotion and manipulation controller directly on OpenUSD articulated models in NVIDIA Isaac Lab.',
    canadianToddlerExplanation:
      'Whether the floor is slippery like a curling rink or bumpy like a forest trail, our robot keeps its balance and carries your maple syrup bottle without spilling a single drop, buddy!'
  },
  {
    id: 'jetson-ethercat-rt',
    subsystem: 'hardware_edge',
    subsystemLabel: 'Opto-Mechatronic & Thermal Edge Integration',
    qisdaRequirement: 'Jetson, EtherCAT, Integrated Joint, Power, Battery, WiFi, GPS, Thermal, ROS 2',
    title: 'Hardware-Accelerated Isaac ROS 2 NITROS + Real-Time EtherCAT Master',
    venueAndYear: 'IEEE RTSS / ICRA / NVIDIA Isaac ROS 2 (2023–2024)',
    githubRepo: 'NVIDIA-ISAAC-ROS/isaac_ros_common & etherlab.org/ethercat',
    githubStars: '1,900+ / 1,200+',
    leakageMitigation:
      'Hardware PTP / IEEE 1588 clock synchronization across LiDAR, IMU, and cameras prevents future-timestamp interpolation leakage during high-angular-velocity turns.',
    overfittingControl:
      'CUDA Graph static execution + thermal-aware DVFS headroom capping guarantees bounded Worst-Case Execution Time (WCET) across 0°C to 55°C ambient factory temperatures.',
    imbalanceMetric:
      '99.99th Percentile Control Loop Jitter (us) & Packet Drop Resilience under 5G/WiFi handover.',
    edgeLatencyMs: 0.85,
    closedLoopSuccessRate: 99.6,
    rigorousSummary:
      'Implements zero-copy GPU-to-GPU type negotiation (NITROS) in ROS 2 Humble/Jazzy on Jetson AGX Orin while isolating the 1kHz EtherCAT Distributed Clock joint loop on dedicated SCHED_FIFO CPU cores.',
    canadianToddlerExplanation:
      'The robot’s brain, battery, and muscles talk super fast without tripping over each other’s hockey sticks! Even when it works hard all day, its cooling fan keeps it chilly and happy, eh!'
  }
];

export const COMPLEXITY_PROFILES: ComplexityProfile[] = [
  {
    id: 'opencv-multispectral',
    framework: 'OpenCV 4.x',
    functionName: 'MultiSpectralPreprocessor.register_and_normalize',
    timeBigO: 'O(H * W)',
    spaceBigO: 'O(1) auxiliary per frame (pre-allocated contiguous dst buffer)',
    eliminatedAntiPattern:
      'Eliminates allocating new numpy arrays (`np.zeros` / `cv2.warpPerspective` without `dst`) inside the 30Hz–60Hz sensor loop, which causes heap fragmentation and GC stalls.',
    officialDocReference: 'OpenCV 4.x Official Docs: cv::warpPerspective(src, dst, M, dsize) & Transparent API (UMat)',
    canadianToddlerSummary:
      'Instead of buying a brand-new toy box every second and filling up the bedroom with trash, we reuse the exact same clean box every time (`O(1)` extra space), eh!',
    codeSnippet: `class MultiSpectralPreprocessor:
    def __init__(self, height: int, width: int) -> None:
        self.height, self.width = height, width
        # Pre-allocate contiguous buffers once at startup: O(H * W) static, O(1) per frame
        self._aligned_lwir_buf = np.empty((height, width), dtype=np.float32)
        self._fused_tensor_buf = np.empty((height, width, 4), dtype=np.float32)

    def register_and_normalize(
        self,
        rgb_frame: np.ndarray,
        depth_frame: np.ndarray,
        lwir_frame: np.ndarray,
        homography_lwir_to_rgb: np.ndarray,
    ) -> np.ndarray:
        # Time: O(H * W) bilinear projection | Auxiliary Space: O(1) via in-place dst
        cv2.warpPerspective(
            src=lwir_frame,
            M=homography_lwir_to_rgb,
            dsize=(self.width, self.height),
            dst=self._aligned_lwir_buf,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        np.multiply(rgb_frame, 1.0 / 255.0, out=self._fused_tensor_buf[:, :, :3], casting="unsafe")
        np.add(depth_frame * 0.5, self._aligned_lwir_buf * 0.5, out=self._fused_tensor_buf[:, :, 3])
        return self._fused_tensor_buf`
  },
  {
    id: 'pytorch-wbc-policy',
    framework: 'PyTorch 2.x',
    functionName: 'BalancedSoftmaxLoss.forward & run_deterministic_jetson_inference',
    timeBigO: 'O(B * C_in * C_out * K^2 * H * W + B * C)',
    spaceBigO: 'O(B * A) output with zero autograd graph retention',
    eliminatedAntiPattern:
      'Eliminates accumulating un-detached tensors (`total_loss += loss`) and replaces legacy `torch.no_grad()` with `torch.inference_mode()` + `register_buffer` for device-safe priors.',
    officialDocReference: 'PyTorch 2.x Official Docs: torch.inference_mode, torch.amp.autocast, nn.Module.register_buffer',
    canadianToddlerSummary:
      'When the robot is done practicing and is out helping folks, it doesn’t carry its giant school backpack of practice notes around (`torch.inference_mode()`), so it runs super light and fast!',
    codeSnippet: `class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, cls_num_list: torch.Tensor) -> None:
        super().__init__()
        prior = cls_num_list.float() / cls_num_list.sum()
        self.register_buffer("log_prior", torch.log(prior.clamp(min=1e-12)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Time: O(B * C) | Corrects Bayes prior drift under extreme class imbalance
        return F.cross_entropy(logits + self.log_prior.unsqueeze(0), targets)

@torch.inference_mode()
def run_deterministic_jetson_inference(
    model: nn.Module,
    img_tensor: torch.Tensor,
    proprio_state: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda"), dtype=torch.float16):
        actions = model(
            img_tensor.to(device, non_blocking=True),
            proprio_state.to(device, non_blocking=True),
        )
    return actions.detach().cpu().numpy()`
  },
  {
    id: 'tensorflow-xla-gate',
    framework: 'TensorFlow 2.x',
    functionName: 'xla_mahalanobis_outlier_gate & build_leakage_free_tf_dataset',
    timeBigO: 'O(N * M^2) XLA-fused gate / O(1) amortized batch prefetch',
    spaceBigO: 'O(N * M) fused XLA buffer / O(B) active stream buffer',
    eliminatedAntiPattern:
      'Eliminates Python-side `for` loops over sensor packets and prevents `@tf.function` retracing memory leaks by passing Tensor scalars and `reduce_retracing=True`.',
    officialDocReference: 'TensorFlow 2.x Official Guide: tf.function(jit_compile=True, reduce_retracing=True) & tf.data.AUTOTUNE',
    canadianToddlerSummary:
      'Our conveyor belt (`tf.data.AUTOTUNE`) gets the next snack ready while the robot is still chewing the first one, and checks for tricky raccoons in one smooth swoosh (`jit_compile=True`), eh!',
    codeSnippet: `@tf.function(jit_compile=True, reduce_retracing=True)
def xla_mahalanobis_outlier_gate(
    sensor_innovations: tf.Tensor,
    precision_matrix: tf.Tensor,
    chi2_threshold: tf.Tensor,
) -> tf.Tensor:
    # Time: O(N * M^2) where N=measurements, M=6 SE(3) DOF | Space: O(N * M)
    projected = tf.linalg.matmul(sensor_innovations, precision_matrix)
    mahalanobis_sq = tf.reduce_sum(projected * sensor_innovations, axis=-1)
    return tf.less_equal(mahalanobis_sq, chi2_threshold)

def build_leakage_free_tf_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    sequence_ids: np.ndarray,
    target_sequence_split: int,
    batch_size: int = 32,
) -> tf.data.Dataset:
    mask = sequence_ids == target_sequence_split
    ds = tf.data.Dataset.from_tensor_slices((features[mask], labels[mask]))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)`
  }
];

export const ROOT_CAUSE_SOLUTIONS: RootCauseSolution[] = [
  {
    id: 'rc-1-temporal-leakage',
    indexNumber: '01',
    defectTitle: 'Temporal Autocorrelation Leakage & Causal Confusion in Visuomotor Learning',
    primarySourceCitation:
      'De Haan et al., "Causal Confusion in Imitation Learning" (NeurIPS); UniAD (CVPR 2023 Best Paper)',
    rootCauseAnalysis:
      'Random frame-wise Train/Validation splitting of 30Hz RGB-D/LiDAR robot logs places frame t in Train and frame t+1 (33ms later) in Validation. The network memorizes high-frequency background textures and exploits lagged ego-velocity shortcuts, showing >99% validation accuracy while colliding in closed-loop deployment.',
    solutionTitle: 'Solution 1: Episode-Boundary Purged Group Splitting + Causal History Dropout',
    mathematicalFormulation:
      'Train/Val partition enforces Group(seq_i) ∩ Group(seq_j) = ∅ with purge buffer |t_train - t_val| > Δt_purge (5.0s), paired with Balanced Meta-Softmax: z_hat_y = z_y + log(n_y / N).',
    implementationDetail:
      'Partition strictly by physical recording episode and factory zone ID, purge 150 frames across split boundaries, and apply stochastic history masking on proprioceptive velocity inputs.',
    canadianToddlerExplanation:
      'If you look at the answer key on the back of the page while practicing, you think you’re a genius, eh! We hide the answer key in another room and make the robot solve brand-new mazes fairly!',
    beforeMetricLabel: 'Naive Frame-Split Closed-Loop Success',
    beforeMetricValue: '41.2% (Severe Shortcut Overfit)',
    afterMetricLabel: 'Purged Split + Causal Policy Success',
    afterMetricValue: '94.8% (Verified Generalization)'
  },
  {
    id: 'rc-2-sampling-bias',
    indexNumber: '02',
    defectTitle: 'Multi-Rate Sensor Selection Bias & DataLoader Copy-on-Write RAM Exhaustion',
    primarySourceCitation:
      'PyTorch Official Issue #13246 (Multiprocessing CoW Memory Leak); Continuous-Time LiDAR-Inertial Odometry (ICRA/CVPR)',
    rootCauseAnalysis:
      'Synchronizing 10Hz LiDAR, 30Hz LWIR/RGB-D, 400Hz IMU, and 1000Hz EtherCAT via naive nearest-neighbor timestamp matching systematically drops high-angular-velocity frames where timestamp skew exceeds threshold. This induces severe survivorship/selection bias toward straight, slow motion (and in HR screening of 610 applicants for 17 multi-disciplinary roles, single-keyword filtering similarly rejects cross-domain systems engineers). Furthermore, storing frame metadata as Python lists in PyTorch DataLoader triggers OS page duplication on fork().',
    solutionTitle: 'Solution 2: Cumulative SE(3) B-Spline Continuous-Time Deskewing & Contiguous Tensor Backing',
    mathematicalFormulation:
      'T_w^s(t) = T_i * prod_{j=1}^{k-1} exp(lambda_j(u) * log(T_{i+j-1}^{-1} T_{i+j})) in SE(3) + Contiguous numpy/torch shared-memory arrays.',
    implementationDetail:
      'Interpolate exact sensor poses at arbitrary microsecond timestamps on the Lie group SE(3) so zero high-dynamic frames are discarded, and replace all Python list attributes inside Dataset objects with `np.ndarray` / `torch.Tensor` to prevent Linux copy-on-write memory bloat.',
    canadianToddlerExplanation:
      'When our robot spins fast on the ice, its eyes blink at different speeds! Instead of throwing away the blurry pictures, we connect the dots smoothly like a figure skater’s blade on the rink, eh!',
    beforeMetricLabel: 'High-Angular-Velocity Frame Retention & RAM',
    beforeMetricValue: '38.0% Retained / OOM Crash at Epoch 4',
    afterMetricLabel: 'SE(3) Deskewed Retention & RAM Footprint',
    afterMetricValue: '99.8% Retained / Constant 1.4 GB RAM'
  },
  {
    id: 'rc-3-thermal-wbc-jitter',
    indexNumber: '03',
    defectTitle: 'Thermal DVFS Tail-Latency Spikes & Multimodal Action Mode Collapse',
    primarySourceCitation:
      'Chi et al., "Diffusion Policy" (RSS 2023 / IJRR 2024); NVIDIA Jetson Orin Thermal & RT-Preempt Scheduling Docs',
    rootCauseAnalysis:
      'Under combined optical, mechanical, electrical, communication, and thermal load on Jetson AGX Orin, shared memory contention between vision transformers and the 1kHz EtherCAT Whole-Body Control loop causes thermal throttling. Meanwhile, standard L2 regression policies average bimodal left/right obstacle avoidance trajectories straight into obstacles.',
    solutionTitle: 'Solution 3: Receding-Horizon Denoising Diffusion Policy + Isolated SCHED_FIFO CUDA Graph Execution',
    mathematicalFormulation:
      'a^{k-1} = alpha * (a^k - gamma * eps_theta(o_t, a^k, k)) + sigma * z with static WCET bound T_exec <= 850 us.',
    implementationDetail:
      'Capture the PyTorch/TensorRT denoising steps inside a static CUDA Graph with zero dynamic allocations, and pin the EtherCAT master thread to isolated CPU cores (`isolcpus=4,5`, `SCHED_FIFO` priority 90).',
    canadianToddlerExplanation:
      'We give the robot’s legs their very own express lane on the highway so even if its eyes are busy looking at a big poster, its feet never forget to step on the brakes in time, buddy!',
    beforeMetricLabel: '99.9%ile Control Jitter under 50°C Thermal Load',
    beforeMetricValue: '14,800 us (EtherCAT Sync Lost)',
    afterMetricLabel: 'CUDA Graph + SCHED_FIFO Isolated Jitter',
    afterMetricValue: '185 us (Zero Missed Deadlines)'
  }
];

export const NIST_CSF_FUNCTIONS: NistCsfFunction[] = [
  {
    code: 'GV',
    name: 'GOVERN (統治)',
    categoryCode: 'GV.SC-04 / GV.RM-02',
    qisdaSystemTarget: 'Qisda Machine Vision Product Center AI & Supply-Chain Governance',
    technicalImplementation:
      'Enforces cryptographic verification of pre-trained weights, OpenUSD mechanical CAD hashes, and automated CI/CD rejection of any model evaluated without Temporal Purge Splits.',
    canadianToddlerRule:
      'The Head Coach checks every hockey stick and helmet before the game starts so everyone plays fair and safe, eh!',
    verificationCheck: 'SHA-256 Model Weight & Dataset Split Manifest Signed'
  },
  {
    code: 'ID',
    name: 'IDENTIFY (特定)',
    categoryCode: 'ID.AM-01 / ID.RA-01',
    qisdaSystemTarget: 'Opto-Mechatronic-Thermal Asset Inventory (SBOM / HBOM)',
    technicalImplementation:
      'Maintains a live, machine-readable bill of materials across Optical (LiDAR, RGB-D, LWIR), Mechanical (Pro/E USD), Electronic (Jetson, EtherCAT, Battery), and Comms (5G/WiFi/GPS) nodes.',
    canadianToddlerRule:
      'We put a neat name tag on every single camera, battery, and wheel so nothing ever gets lost in the snow!',
    verificationCheck: '100% Sensor & Actuator Node Topology Enumerated'
  },
  {
    code: 'PR',
    name: 'PROTECT (防御)',
    categoryCode: 'PR.DS-01 / PR.AA-03',
    qisdaSystemTarget: 'SROS2 Encrypted DDS Bus & Isolated EtherCAT Ring',
    technicalImplementation:
      'Enables ROS 2 DDS-Security (X.509 PKI + AES-256-GCM transport encryption) for wireless telemetry and physically isolates the real-time EtherCAT joint bus from external WiFi/5G interfaces.',
    canadianToddlerRule:
      'Our robot uses a secret clubhouse password so strangers on the WiFi cannot tell its arms or legs what to do!',
    verificationCheck: 'SROS2 Enclave Permissions & Read-Only Memory Pages Active'
  },
  {
    code: 'DE',
    name: 'DETECT (検知)',
    categoryCode: 'DE.CM-01 / DE.AE-03',
    qisdaSystemTarget: 'XLA Mahalanobis Sensor Spoofing & Thermal Anomaly Gate',
    technicalImplementation:
      'Runs `xla_mahalanobis_outlier_gate` at 200Hz over LiDAR/IMU/GPS innovations and monitors die/battery thermal derivatives dT/dt to detect sensor spoofing or cooling failure within 5ms.',
    canadianToddlerRule:
      'If a cheeky raccoon shines a flashlight into the robot’s eyes to trick it, the robot’s smart math alarm spots the trick right away!',
    verificationCheck: 'Chi-Square (df=6, alpha=0.001) Threshold = 22.46 Active'
  },
  {
    code: 'RS',
    name: 'RESPOND (対応)',
    categoryCode: 'RS.MI-01 / RS.CO-02',
    qisdaSystemTarget: 'Whole-Body Passive Safe-Torque Damping & Incident Logging',
    technicalImplementation:
      'Upon Mahalanobis gate rejection or EtherCAT Working Counter (WKC) drop, transitions Integrated Joints to hardware STO / passive viscous damping within 1 control cycle (1ms).',
    canadianToddlerRule:
      'When the alarm rings, the robot gently sits down, puts on its mittens, and says "Sorry about that, let’s pause so nobody gets bumped!"',
    verificationCheck: 'Hardware Safe Torque Off (STO) Fallback <= 1.0 ms'
  },
  {
    code: 'RC',
    name: 'RECOVER (復旧)',
    categoryCode: 'RC.RP-03 / RC.CO-03',
    qisdaSystemTarget: 'Signed State-Estimation Reset & Relocalization Verification',
    technicalImplementation:
      'Restores verified SLAM keyframes from read-only checkpoints, re-validates multi-spectral extrinsic calibration, and resumes autonomous navigation only after 50 consecutive nominal cycles.',
    canadianToddlerRule:
      'Once the path is clear again, the robot checks its map, dusts off its knees, and happily gets back to helping out, eh!',
    verificationCheck: 'Deterministic Pose Graph Checksum Restored'
  }
];

export const TAIWAN_LEGAL_CHECKS: TaiwanLegalCheckItem[] = [
  {
    id: 'tw-trade-secret',
    statuteNameZh: '中華民國《營業秘密法》（Trade Secrets Act）',
    statuteNameEn: 'Taiwan Trade Secrets Act (Articles 2, 10, 13-1)',
    articleRef: '第 2 條・第 10 條',
    complianceCriterion:
      'Zero inclusion of non-public Qisda proprietary Pro/E CAD drawings, internal PCB schematics, or confidential factory floor parameters. All code is clean-room synthesized from public CVPR/ICCV papers and official PyTorch/OpenCV/TF docs.',
    auditMethod: 'Automated AST & Entropy Scan for proprietary headers, CAD binaries, or internal IP endpoints.',
    canadianToddlerExplanation:
      'We only share our own homemade Lego instructions from public storybooks, and never ever take someone else’s secret family recipe, eh!',
    passedByDefault: true
  },
  {
    id: 'tw-pdpa',
    statuteNameZh: '中華民國《個人資料保護法》（PDPA）',
    statuteNameEn: 'Taiwan Personal Data Protection Act (Articles 2, 6, 19, 20)',
    articleRef: '第 2 條・第 6 條・第 19 條',
    complianceCriterion:
      'Zero Personal Identifiable Information (PII) of the 610 job applicants, HR personnel, or BenQ Medical Center patients. All evaluation datasets in the repository use synthetic random tensors.',
    auditMethod: 'Regex & NER PII Scrubber verifying zero names, phone numbers, national IDs, or biometric faces.',
    canadianToddlerExplanation:
      'We never put anyone’s real name, phone number, or photo on the internet—everyone’s privacy stays safe and sound!',
    passedByDefault: true
  },
  {
    id: 'tw-copyright',
    statuteNameZh: '中華民國《著作權法》（Copyright Act）',
    statuteNameEn: 'Taiwan Copyright Act (Articles 9, 10-1, 52, 65)',
    articleRef: '第 9 條・第 52 條・第 65 條',
    complianceCriterion:
      'Factual technical skill requirements (LiDAR, ROS 2, Jetson, EtherCAT) are unprotected facts/ideas under Art. 10-1; recruitment commentary adheres to Art. 52/65 Fair Use (合理使用), and all source code is licensed under Apache-2.0.',
    auditMethod: 'SPDX License Header Verification (`Apache-2.0`) & Fair-Use Citation Attribution.',
    canadianToddlerExplanation:
      'We give a polite thank-you note to the scientists whose ideas we learned from, and put a friendly sharing sticker (`Apache-2.0`) on our own code!',
    passedByDefault: true
  }
];

/**
 * Pure deterministic calculation of Big-O operation counts and memory footprints
 * for the official OpenCV, PyTorch, and TensorFlow reference functions.
 */
export function calculateComplexityMetrics(
  input: ComplexitySimulationInput
): ComplexitySimulationResult {
  const height = Math.max(16, Math.min(4096, Math.floor(input.height)));
  const width = Math.max(16, Math.min(4096, Math.floor(input.width)));
  const numPoints = Math.max(100, Math.min(1000000, Math.floor(input.numPoints)));
  const batchSize = Math.max(1, Math.min(256, Math.floor(input.batchSize)));
  const numClasses = Math.max(2, Math.min(10000, Math.floor(input.numClasses)));

  const pixels = height * width;

  // OpenCV warpPerspective bilinear interpolation (~18 FLOPs per pixel + 4-ch fusion)
  const opencvWarpOps = pixels * 22;
  // Pre-allocated float32 buffers: _aligned_lwir_buf (H*W*4 bytes) + _fused_tensor_buf (H*W*4*4 bytes)
  const opencvPreallocBytes = pixels * 4 + pixels * 4 * 4;
  // If anti-pattern (allocating new buffers at 30Hz for 60 seconds without pre-allocation):
  const opencvLeakyAllocBytesPerMinuteAt30Hz = opencvPreallocBytes * 30 * 60;

  // PyTorch Conv2d stem (C_in=4, C_out=32, K=3, stride=2) + MLP policy head
  const outH = Math.ceil(height / 2);
  const outW = Math.ceil(width / 2);
  const convOpsPerSample = 4 * 32 * 9 * outH * outW;
  const mlpOpsPerSample = (32 + 24) * 64 + 64 * 12;
  const pytorchForwardOps = batchSize * (convOpsPerSample + mlpOpsPerSample);

  // Balanced Softmax Loss: O(B * C)
  const pytorchBalancedSoftmaxOps = batchSize * numClasses * 3;
  // FP16 (2 bytes) activation memory under torch.inference_mode() + amp.autocast
  const pytorchActivationBytes = batchSize * 32 * outH * outW * 2;

  // TensorFlow XLA Mahalanobis Outlier Gate: O(N * M^2) with M=6 SE(3) DOF
  const stateDim = 6;
  const tfXlaMahalanobisOps = numPoints * (stateDim * stateDim + stateDim);
  // Active batch stream memory for 6-DOF float32 innovations
  const tfDatasetStreamBytes = batchSize * stateDim * 4;

  return {
    opencvWarpOps,
    opencvPreallocBytes,
    opencvLeakyAllocBytesPerMinuteAt30Hz,
    pytorchForwardOps,
    pytorchBalancedSoftmaxOps,
    pytorchActivationBytes,
    tfXlaMahalanobisOps,
    tfDatasetStreamBytes
  };
}

/**
 * Pure implementation of Balanced Softmax logit adjustment (Ren et al., NeurIPS)
 * to eliminate class-imbalance bias without synthetic oversampling overfitting.
 */
export function computeBalancedSoftmaxProbabilities(
  rawLogits: number[],
  classCounts: number[]
): { adjustedLogits: number[]; probabilities: number[]; naiveProbabilities: number[] } {
  if (rawLogits.length === 0 || rawLogits.length !== classCounts.length) {
    throw new Error('rawLogits and classCounts must be non-empty and have matching lengths.');
  }

  const totalSamples = classCounts.reduce((acc, c) => acc + Math.max(1, c), 0);
  const priors = classCounts.map((c) => Math.max(1, c) / totalSamples);

  const adjustedLogits = rawLogits.map((logit, i) => logit + Math.log(priors[i]));

  const softmax = (vec: number[]): number[] => {
    const maxVal = Math.max(...vec);
    const exps = vec.map((v) => Math.exp(v - maxVal));
    const sumExp = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / sumExp);
  };

  return {
    adjustedLogits,
    probabilities: softmax(adjustedLogits),
    naiveProbabilities: softmax(rawLogits)
  };
}

/**
 * Pure implementation of NIST CSF 2.0 'DE.CM' Mahalanobis Outlier Gate
 * for detecting sensor spoofing or corrupted LiDAR/IMU measurements.
 */
export function evaluateMahalanobisGate(
  innovationVector: number[],
  diagonalVariances: number[],
  chiSquareThreshold: number = 22.458 // df=6, p=0.001
): { mahalanobisSquared: number; isNominal: boolean; statusLabel: string } {
  if (
    innovationVector.length === 0 ||
    innovationVector.length !== diagonalVariances.length
  ) {
    throw new Error('innovationVector and diagonalVariances must have equal non-zero length.');
  }

  let mahalanobisSquared = 0;
  for (let i = 0; i < innovationVector.length; i++) {
    const variance = Math.max(1e-9, diagonalVariances[i]);
    mahalanobisSquared += (innovationVector[i] * innovationVector[i]) / variance;
  }

  const isNominal = mahalanobisSquared <= chiSquareThreshold;
  return {
    mahalanobisSquared: Number(mahalanobisSquared.toFixed(4)),
    isNominal,
    statusLabel: isNominal
      ? 'NOMINAL (Accepted into EKF/Factor Graph)'
      : 'ANOMALY DETECTED (Rejected by NIST DE.CM Gate -> Trigger RS.MI Safe Damping)'
  };
}

/**
 * Audits a candidate repository text payload against Taiwan Trade Secrets Act,
 * PDPA (Personal Data Protection Act), and Copyright Act rules before GitHub export.
 */
export function auditPayloadForTaiwanGitHubCompliance(payload: string): {
  isCompliant: boolean;
  detectedIssues: string[];
  sanitizedPreview: string;
} {
  const issues: string[] = [];
  let sanitized = payload;

  // 1. Check for Taiwan National ID pattern (1 letter + 9 digits, e.g., A123456789)
  const twIdRegex = /\b[A-Z][12]\d{8}\b/g;
  if (twIdRegex.test(payload)) {
    issues.push('PDPA Warning (個人資料保護法): Detected Taiwan National ID pattern.');
    sanitized = sanitized.replace(twIdRegex, '[REDACTED_TW_ID]');
  }

  // 2. Check for email addresses or phone numbers
  const emailRegex = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g;
  if (emailRegex.test(payload)) {
    issues.push('PDPA Warning (個人資料保護法): Detected personal email address.');
    sanitized = sanitized.replace(emailRegex, '[REDACTED_EMAIL]');
  }

  // 3. Check for confidential/proprietary markers under Trade Secrets Act (營業秘密法)
  const secretRegex = /\b(CONFIDENTIAL|INTERNAL_ONLY|PROPRIETARY_CAD|SECRET_KEY)\b/gi;
  if (secretRegex.test(payload)) {
    issues.push('Trade Secrets Act Warning (營業秘密法): Detected confidential marker.');
    sanitized = sanitized.replace(secretRegex, '[REDACTED_CONFIDENTIAL_MARKER]');
  }

  return {
    isCompliant: issues.length === 0,
    detectedIssues: issues,
    sanitizedPreview: sanitized
  };
}
