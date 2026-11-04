import torch
import torch.nn as nn
import numpy as np
import math
from PIL import Image
import torchvision.transforms as T
import gsplat
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy
import pycolmap
from pathlib import Path
import time
import viser
import viser.transforms as tf
from plyfile import PlyData, PlyElement

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pycolmap.logging.minloglevel = pycolmap.logging.WARNING


def run_COLMAP(cfg, seq_id):
    image_dir = Path(cfg.path.dir_image_panomap) / seq_id
    output_dir = Path(cfg.path.dir_gs) / seq_id
    output_dir.mkdir(parents=True, exist_ok=True)

    database_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"

    if not database_path.exists():
        # Feature Extraction
        pycolmap.extract_features(
            database_path=database_path,
            image_path=image_dir,
            camera_mode=pycolmap.CameraMode.PER_FOLDER,
            camera_model="PINHOLE",
        )

        # Feature Matching
        pycolmap.match_exhaustive(database_path=database_path) # try later if sequential matching works

    if not sparse_dir.exists():
        # 3D SfM (Sparse Reconstruction)
        reconstructions = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=image_dir,
            output_path=sparse_dir
        )


def visualize_COLMAP(cfg, seq_id):
    server = viser.ViserServer(host="0.0.0.0", port=8080)

    sparse_dir = Path(cfg.path.dir_gs) / seq_id / "sparse" / "0"
    rec = pycolmap.Reconstruction(sparse_dir)

    points3D = rec.points3D
    xyz = np.array([p.xyz for p in points3D.values()])
    colors = np.array([p.color / 255.0 for p in points3D.values()])

    server.scene.add_point_cloud(
        name="/colmap/points",
        points=xyz,
        colors=colors,
        point_size=0.03,
    )

    # visualize camera frustums
    for img_id, img in rec.images.items():
        world_from_cam = img.cam_from_world().inverse()
        pos = world_from_cam.translation
        # Rererun과 달리 viser는 wxyz quaternion 지원
        quat = tf.SO3.from_matrix(world_from_cam.rotation.matrix()).wxyz

        server.scene.add_camera_frustum(
            name=f"/colmap/cameras/{img.name}",
            fov=1.2,
            aspect=1920 / 1080,
            scale=0.15,
            position=pos,
            wxyz=quat,
        )

    print("Viser server running: waiting for local browser connection...")
    while True:
        time.sleep(1)


def camera_to_K(cam):
    # COLMAP param layouts differ per camera model (distortion is ignored here)
    p = cam.params
    name = str(cam.model.name) if hasattr(cam.model, "name") else str(cam.model)
    name = name.split(".")[-1]
    if name in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx = fy = p[0]
        cx, cy = p[1], p[2]
    else:  # PINHOLE, OPENCV, FULL_OPENCV, ...: [fx, fy, cx, cy, ...]
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def parse_colmap(rec, image_dir):
    images_dict = rec.images
    cameras_dict = rec.cameras
    
    viewmats_list = []
    gt_images_list = []
    Ks_list = []
    
    # Sort image IDs based on image file names to maintain consistent order
    sorted_image_ids = sorted(images_dict.keys(), key=lambda k: images_dict[k].name)
    
    for img_id in sorted_image_ids:
        img_obj = images_dict[img_id]
        
        # 1) Camera-to-World -> World-to-Camera (W2C)
        w2c_mat = np.eye(4, dtype=np.float32)
        w2c_mat[:3, :3] = img_obj.cam_from_world().rotation.matrix()
        w2c_mat[:3, 3] = img_obj.cam_from_world().translation
        viewmats_list.append(w2c_mat)
        
        # 2) Load actual image and convert to Tensor (H, W, 3) [0.0, 1.0]
        img_path = image_dir / img_obj.name
        pil_img = Image.open(img_path).convert("RGB")
        img_np = np.array(pil_img, dtype=np.float32) / 255.0
        gt_images_list.append(img_np)

        cam = cameras_dict[img_obj.camera_id]
        K_i = camera_to_K(cam)
        # rescale intrinsics if the image resolution differs from the COLMAP camera
        h, w = img_np.shape[:2]
        K_i[0, :] *= w / cam.width
        K_i[1, :] *= h / cam.height
        Ks_list.append(K_i)
    
    viewmats_tensor = torch.tensor(np.array(viewmats_list), dtype=torch.float32, device=device)
    gt_images_tensor = torch.tensor(np.array(gt_images_list), dtype=torch.float32, device=device)
    
    # 3) Per-view intrinsic matrices (N, 3, 3)
    K_tensor = torch.tensor(np.array(Ks_list), dtype=torch.float32, device=device)
    
    # 4) Sparse Point Cloud (used as initial Gaussian positions for gsplat)
    points3D = rec.points3D
    xyz_list = [p.xyz for p in points3D.values()]
    colors_list = [p.color / 255.0 for p in points3D.values()] # Initial RGB values
    
    means3D_tensor = torch.tensor(np.array(xyz_list), dtype=torch.float32, device=device)
    init_colors_tensor = torch.tensor(np.array(colors_list), dtype=torch.float32, device=device)
    
    return viewmats_tensor, gt_images_tensor, K_tensor, means3D_tensor, init_colors_tensor


def train_gs(cfg, seq_id):
    sparse_dir = Path(cfg.path.dir_gs) / seq_id / "sparse" / "0"
    image_dir = Path(cfg.path.dir_image_panomap) / seq_id

    rec = pycolmap.Reconstruction(sparse_dir)

    viewmats, gt_images, K, init_means3D, init_colors = parse_colmap(rec, image_dir)
    
    num_views, H, W, _ = gt_images.shape
    num_points = init_means3D.shape[0]
    print(f"Load complete: total {num_views} camera views, resolution ({W}x{H})")
    print(f"Initial 3D points: {num_points}")

    ### 3D Gaussian initialization for gsplat (using COLMAP sparse points as initial positions)
    means3D = nn.Parameter(init_means3D.clone())
    colors = nn.Parameter(torch.logit(init_colors.clamp(0.02, 0.98)))

    # Quaternion rotation (initialized as identity quaternions)
    init_quats = torch.zeros((num_points, 4), device=device)
    init_quats[:, 0] = 1.0
    quats = nn.Parameter(init_quats)

    # Scale (log space), initialized from mean distance to 3 nearest neighbours
    dists = torch.cdist(init_means3D, init_means3D) if num_points <= 20000 else None
    if dists is not None:
        dists.fill_diagonal_(float("inf"))
        knn = dists.topk(3, largest=False).values.mean(dim=-1)
    else:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(init_means3D.cpu().numpy()).query(init_means3D.cpu().numpy(), k=4)
        knn = torch.tensor(d[:, 1:].mean(axis=-1), dtype=torch.float32, device=device)
    knn = knn.clamp(min=1e-6)
    scales = nn.Parameter(torch.log(knn)[:, None].repeat(1, 3))

    # Opacity (defined in logit space)
    opacities = nn.Parameter(torch.logit(torch.ones((num_points, 1), device=device) * 0.5))

    # Scene extent = radius of the camera centers (used for lr and densification thresholds)
    cam_centers = torch.linalg.inv(viewmats)[:, :3, 3]
    scene_scale = float((cam_centers - cam_centers.mean(0)).norm(dim=-1).max().item())
    scene_scale = max(scene_scale, 1e-3)

    # One optimizer per parameter (required by gsplat strategies for densification/pruning)
    params = nn.ParameterDict({
        "means": means3D, "scales": scales, "quats": quats,
        "opacities": opacities, "colors": colors,
    })
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "colors": 2.5e-3}

    # Resume from checkpoint if it exists (the Gaussian count may differ from the init due to densification)
    fp_gs_checkpoint = Path(cfg.path.dir_gs) / seq_id / "gs_checkpoint.pt"
    ckpt = None
    if fp_gs_checkpoint.exists():
        ckpt = torch.load(fp_gs_checkpoint, map_location=device)
        # new format: ckpt["params"]; legacy format: flat tensors with 'means3D' instead of 'means'
        saved = ckpt.get("params", ckpt)
        saved = {("means" if k == "means3D" else k): v for k, v in saved.items()}
        for k in params.keys():
            params[k] = nn.Parameter(saved[k].to(device))
        print(f"Resuming from {fp_gs_checkpoint}: epoch {ckpt.get('epoch', 0)}, step {ckpt.get('step', 0)}, "
              f"{len(params['means'])} Gaussians")

    optimizers = {k: torch.optim.Adam([params[k]], lr=lr) for k, lr in lrs.items()}
    # legacy checkpoints have no optimizer/strategy/scheduler state: warm start only
    full_state = ckpt is not None and "optimizers" in ckpt
    if full_state:
        for k, opt in optimizers.items():
            opt.load_state_dict(ckpt["optimizers"][k])

    epochs = 1000
    total_steps = epochs * num_views

    # Densification (clone/split) and pruning (low opacity / too big) from gsplat
    strategy = DefaultStrategy(
        prune_opa=0.005,
        grow_grad2d=0.0002,
        grow_scale3d=0.01,
        prune_scale3d=0.1,
        refine_start_iter=min(500, total_steps // 10),
        refine_stop_iter=int(total_steps),
        refine_every=100,
        reset_every=3000,
        verbose=True,
    )
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    means_sched = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / total_steps)
    )

    step, start_epoch = 0, 0
    if full_state:
        means_sched.load_state_dict(ckpt["scheduler"])
        step, start_epoch = ckpt["step"], ckpt["epoch"]

    def save_state(epoch_done):
        save_checkpoint(fp_gs_checkpoint, params, optimizers, strategy_state, means_sched, K, epoch_done, step)

    ### Training loop for gsplat
    print("\n--- Start gsplat Training ---")

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        total_loss = 0.0
        
        # Iterate over all camera views for the current epoch
        for idx in torch.randperm(num_views).tolist():
            viewmat = viewmats[idx]
            gt_img = gt_images[idx]
            
            # Run gsplat CUDA rasterizer
            renders, alphas, meta = rasterization(
                means=params["means"],
                quats=torch.nn.functional.normalize(params["quats"], dim=-1),
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"].squeeze(-1)),
                colors=torch.sigmoid(params["colors"]),
                viewmats=viewmat.unsqueeze(0),
                Ks=K[idx].unsqueeze(0),
                width=W,
                height=H,
                packed=False
            )
            strategy.step_pre_backward(params, optimizers, strategy_state, step, meta)
            
            rendered_img = renders.squeeze(0)
            
            # Compute loss (L1 Loss + pixel error)
            loss = torch.mean(torch.abs(rendered_img - gt_img))
            # penalize needle-like Gaussians (max/min scale ratio above 10)
            s_exp = torch.exp(params["scales"])
            loss = loss + 0.01 * torch.relu(s_exp.max(dim=-1).values / s_exp.min(dim=-1).values - 10.0).mean()
            
            loss.backward()
            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)
            means_sched.step()
            strategy.step_post_backward(params, optimizers, strategy_state, step, meta, packed=False)
            step += 1
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_views
        elapsed = time.time() - start_time
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:03d}/{epochs}] - Avg L1 Loss: {avg_loss:.6f} - Gaussians: {len(params['means'])} ({elapsed:.2f}s/epoch)")

        if (epoch + 1) % 25 == 0:
            save_state(epoch + 1)
    print("Training completed!")

    fn_gs_output = "gs_output.ply"
    fp_gs_output = Path(cfg.path.dir_gs) / seq_id / fn_gs_output
    save_ply(fp_gs_output, params["means"], params["colors"], params["scales"], params["quats"], params["opacities"])

    save_state(max(start_epoch, epochs))


def visualize_gs(cfg, seq_id):
    server = viser.ViserServer(host="0.0.0.0", port=8081)

    fn_gs_output = "gs_output.ply"
    fp_gs_output = Path(cfg.path.dir_gs) / seq_id / fn_gs_output
    centers, rgbs, opacities, covariances = load_ply(fp_gs_output)

    server.scene.add_gaussian_splats(
        name="/gaussian_splats",
        centers=centers,
        rgbs=rgbs,
        opacities=opacities,
        covariances=covariances,
    )

    print("Viser server running at http://0.0.0.0:8081")
    while True:
        time.sleep(1)


def load_ply(ply_path):
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    
    centers = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
    rgbs = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=-1) / 255.0
    raw_opacities = vertex['opacity'][:, np.newaxis]
    opacities = 1.0 / (1.0 + np.exp(-raw_opacities)) # apply sigmoid to raw opacities
    scales = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=-1)
    quats = np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=-1)
    quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True) # normalize quaternions
    
    s = np.exp(scales)
    S = np.zeros((len(scales), 3, 3))
    S[:, 0, 0] = s[:, 0]
    S[:, 1, 1] = s[:, 1]
    S[:, 2, 2] = s[:, 2]
    
    R = tf.SO3(quats).as_matrix()
    RS = np.matmul(R, S)
    covariances = np.matmul(RS, RS.transpose(0, 2, 1))
    
    return centers, rgbs, opacities, covariances


def save_ply(filename, means3D, colors, scales, quats, opacities):
    xyz = means3D.detach().cpu().numpy()
    rgb = torch.sigmoid(colors.detach()).cpu().numpy()
    scale = scales.detach().cpu().numpy()
    opacity = opacities.detach().cpu().numpy()
    rot = torch.nn.functional.normalize(quats.detach(), dim=-1).cpu().numpy()

    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
    ]

    rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements['x'], elements['y'], elements['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    elements['red'], elements['green'], elements['blue'] = rgb_uint8[:, 0], rgb_uint8[:, 1], rgb_uint8[:, 2]
    
    elements['opacity'] = opacity.squeeze(-1)
    elements['scale_0'], elements['scale_1'], elements['scale_2'] = scale[:, 0], scale[:, 1], scale[:, 2]
    elements['rot_0'], elements['rot_1'], elements['rot_2'], elements['rot_3'] = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(filename)


def save_checkpoint(filename, params, optimizers, strategy_state, scheduler, K, epoch, step):
    checkpoint = {
        'params': {k: v.detach().cpu() for k, v in params.items()},
        'optimizers': {k: o.state_dict() for k, o in optimizers.items()},
        'strategy_state': strategy_state,
        'scheduler': scheduler.state_dict(),
        'K': K.detach().cpu(),  # (N, 3, 3) per-view
        'epoch': epoch,
        'step': step,
    }

    # write atomically so an interrupted save never corrupts the checkpoint
    tmp = Path(str(filename) + ".tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(filename)

if __name__ == "__main__":
    # visualize_COLMAP(cfg, seq_id)
    # visualize_gs(cfg, seq_id)
    pass