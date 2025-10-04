"""Training script for ToddlerBot locomotion policies using MJX.

This module provides training functionality for ToddlerBot using both JAX (Brax)
and PyTorch (RSL-RL) backends. It supports various locomotion tasks including
walking, crawling, and cartwheel movements with configurable environments.
"""

import os

os.environ["USE_JAX"] = "true"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import argparse
import functools
import importlib
import json
import pkgutil
import shutil
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import gin
import jax
import jax.numpy as jnp
import moviepy.editor as mpy
import mujoco
import numpy as np
import numpy.typing as npt
import torch
import yaml
from brax import base, envs
from brax.io import model
from brax.io.torch import jax_to_torch, torch_to_jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from flax import linen
from mujoco.mjx._src import support
from PIL import Image, ImageDraw, ImageFont
from rsl_rl.utils import store_code_state
from tqdm import tqdm

import wandb
from toddlerbot.locomotion.mjx_config import MJXConfig
from toddlerbot.locomotion.mjx_env import MJXEnv, get_env_class
from toddlerbot.locomotion.on_policy_runner import OnPolicyRunner
from toddlerbot.locomotion.ppo_config import PPOConfig
from toddlerbot.locomotion.rsl_rl_wrapper import RSLRLWrapper
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.misc_utils import (
    dataclass2dict,
    dump_profiling_data,
    # profile,
)

jax.config.update("jax_default_matmul_precision", "highest")
# jax.config.update("jax_debug_nans", True) # this will slow down training significantly

warnings.filterwarnings(
    "ignore",
    message="Brax System, piplines and environments are not actively being maintained.*",
    category=UserWarning,
    module="brax.io.mjcf",
)

warnings.filterwarnings(
    "ignore",
    message="overflow encountered in cast",
    category=RuntimeWarning,
    module="jax._src.interpreters.xla",
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dynamic_import_envs(env_package: str):
    """Import all modules from a specified package for environment registration."""
    """Imports all modules from a specified package.

    This function dynamically imports all modules within a given package, allowing their contents to be accessed programmatically. It is useful for loading environment configurations or plugins from a specified package directory.

    Args:
        env_package (str): The name of the package from which to import all modules.
    """
    package = importlib.import_module(env_package)
    package_path = package.__path__

    # Iterate over all modules in the given package directory
    for _, module_name, _ in pkgutil.iter_modules(package_path):
        full_module_name = f"{env_package}.{module_name}"
        importlib.import_module(full_module_name)


# Call this to import all policies dynamically
dynamic_import_envs("toddlerbot.locomotion")


class Tee:
    """Custom stdout/stderr redirection class for logging output to both console and file."""

    def __init__(self, log_path):
        """Initialize with log file path for dual output."""
        self.terminal = sys.__stdout__  # Use sys.__stdout__ to avoid redirect loops
        self.log = open(log_path, "w", buffering=1)  # Line-buffered
        self.closed = False

    def write(self, message):
        """Write message to both terminal and log file."""
        msg = str(message)
        try:
            self.terminal.write(msg)
            self.terminal.flush()
            if not self.closed:
                self.log.write(msg)
                self.log.flush()
        except (ValueError, OSError):
            self.closed = True

    def flush(self):
        try:
            self.terminal.flush()
            if not self.closed:
                self.log.flush()
        except (ValueError, OSError):
            self.closed = True

    def isatty(self):
        return self.terminal.isatty()

    def fileno(self):
        return self.terminal.fileno()

    def close(self):
        if not self.closed:
            try:
                self.log.close()
            except Exception:
                pass
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


def load_jax_ckpt_to_torch(jax_params):
    """Convert JAX model parameters to PyTorch format for cross-framework compatibility."""
    flat_jax = {}

    def build_tensor(value):
        """Convert a JAX array to a PyTorch tensor."""
        t = torch.tensor(np.array(value).copy(), dtype=torch.float32)
        return t.T if t.ndim > 1 else t

    def add_prefix(d, prefix=""):
        for k, v in d.items():
            new_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                add_prefix(v, new_key)
            else:
                flat_jax[new_key] = build_tensor(v)

    for jax_param in [jax_params[1]["params"], jax_params[2]["params"]]:
        for key, value in jax_param.items():
            if isinstance(value, dict):
                add_prefix(value, prefix=key)
            else:
                flat_jax[key] = build_tensor(value)

    # Map from JAX naming to PyTorch (basic heuristics)
    jax_to_torch_name = {
        "std_logparam.log_value": "log_std",
        "MLP_0.hidden_0.kernel": "actor.0.weight",
        "MLP_0.hidden_0.bias": "actor.0.bias",
        "MLP_0.hidden_1.kernel": "actor.2.weight",
        "MLP_0.hidden_1.bias": "actor.2.bias",
        "MLP_0.hidden_2.kernel": "actor.4.weight",
        "MLP_0.hidden_2.bias": "actor.4.bias",
        "Dense_0.kernel": "actor.6.weight",
        "Dense_0.bias": "actor.6.bias",
        "hidden_0.kernel": "critic.0.weight",
        "hidden_0.bias": "critic.0.bias",
        "hidden_1.kernel": "critic.2.weight",
        "hidden_1.bias": "critic.2.bias",
        "hidden_2.kernel": "critic.4.weight",
        "hidden_2.bias": "critic.4.bias",
        "hidden_3.kernel": "critic.6.weight",
        "hidden_3.bias": "critic.6.bias",
    }

    loaded_dict = {"model_state_dict": {}}
    for jax_key, torch_name in jax_to_torch_name.items():
        if jax_key in flat_jax:
            loaded_dict["model_state_dict"][torch_name] = flat_jax[jax_key]
        else:
            print(f"Missing or unmatched JAX param for PyTorch: {jax_key}-{torch_name}")

    return loaded_dict


def render_video(
    env: MJXEnv,
    states: List[Any],
    video_dir: str,
    video_name: str,
    cameras: List[str] = ["perspective"],
    render_every: int = 2,
    height: int = 360,
    width: int = 640,
):
    """Renders and saves a video of the environment from multiple camera angles.

    Args:
        env (MJXEnv): The environment to render.
        rollout (List[Any]): A list of environment states or actions to render.
        run_name (str): The name of the run, used to organize output files.
        render_every (int, optional): Interval at which frames are rendered from the rollout. Defaults to 2.
        height (int, optional): The height of the rendered video frames. Defaults to 360.
        width (int, optional): The width of the rendered video frames. Defaults to 640.

    Creates:
        A video file for each camera angle ('perspective', 'side', 'top', 'front') and a final concatenated video in a 2x2 grid layout, saved in the 'results' directory under the specified run name.
    """
    # Define paths for each camera's video
    if len(video_dir) > 0:
        os.makedirs(video_dir, exist_ok=True)

    fps = 1.0 / env.dt / render_every
    # Render and write individual camera videos
    camera_clips = []
    for camera in cameras:
        frames = env.render(
            states[::render_every], height=height, width=width, camera=camera
        )
        video = mpy.concatenate_videoclips(
            [mpy.ImageClip(f).set_duration(1 / fps) for f in frames],
            method="compose",
        )
        camera_clips.append(video)

    # Pad with black clips to make 4 total
    duration = camera_clips[0].duration
    if len(camera_clips) > 1:
        while len(camera_clips) < 4:
            camera_clips.append(
                mpy.ColorClip(size=(width, height), color=(0, 0, 0), duration=duration)
            )
        # Arrange in 2x2 grid
        final_video = mpy.clips_array(
            [[camera_clips[0], camera_clips[1]], [camera_clips[2], camera_clips[3]]]
        )
    else:
        final_video = camera_clips[0]

    command_to_render = [
        states[i].info["command"] for i in range(0, len(states), render_every)
    ]
    name_to_render = [
        support.id2name(env.sys, mujoco.mjtObj.mjOBJ_BODY, states[i].info["push_id"])
        for i in range(0, len(states), render_every)
    ]
    push_to_render = [
        states[i].info["push"] for i in range(0, len(states), render_every)
    ]

    def make_text_image(text: str, font_size: int = 18):
        img = Image.new(
            "RGBA", (final_video.w, final_video.h), (0, 0, 0, 0)
        )  # Transparent background
        draw = ImageDraw.Draw(img)
        try:
            # Try to load the preferred font
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except (OSError, IOError):
            try:
                # Fallback to system default font
                font = ImageFont.load_default()
            except (OSError, IOError):
                # Last resort: use default font
                font = None
        draw.text((10, 10), text, font=font, fill="white")
        return img

    name_curr = "torso"
    push_curr = np.zeros(2, dtype=np.float32)

    def make_annotated_frame(t):
        nonlocal name_curr, push_curr
        frame = final_video.get_frame(t)
        i = int(t * fps)
        if i >= len(command_to_render):
            return frame

        if np.linalg.norm(push_to_render[i]) > 0:
            name_curr = name_to_render[i]
            push_curr = push_to_render[i]

        command_str = (
            f"Command: [{', '.join(f'{x:.2f}' for x in command_to_render[i][5:])}]"
        )
        push_str = (
            f"\nPush: [{', '.join(f'{x:.2f}' for x in push_curr)}] at {name_curr}"
        )

        text_img = make_text_image(command_str + push_str)
        composed = Image.alpha_composite(
            Image.fromarray(frame).convert("RGBA"), text_img
        ).convert("RGB")
        return np.array(composed).astype(np.uint8)

    annotated_video = mpy.VideoClip(make_annotated_frame, duration=final_video.duration)
    annotated_video.write_videofile(
        os.path.join(video_dir, f"{video_name}.mp4"), fps=fps
    )


def print_metrics(
    metrics: Dict[str, Any],
    time_elapsed: float,
    num_steps: int,
    num_total_steps: int,
    notes: str = "",
    width: int = 80,
    pad: int = 35,
):
    """Logs and formats metrics for display, including elapsed time and optional step information.

    Args:
        metrics (Dict[str, Any]): A dictionary containing metric names and their corresponding values.
        time_elapsed (float): The time elapsed since the start of the process.
        num_steps (int, optional): The current number of steps completed. Defaults to -1.
        num_total_steps (int, optional): The total number of steps to be completed. Defaults to -1.
        width (int, optional): The width of the log display. Defaults to 80.
        pad (int, optional): The padding for metric names in the log display. Defaults to 35.

    Returns:
        Dict[str, Any]: A dictionary containing the logged data, including time elapsed and processed metrics.
    """
    log_string = f"""{"#" * width}\n"""
    if num_steps >= 0 and num_total_steps > 0:
        title = f" \033[1m Learning steps {num_steps}/{num_total_steps} \033[0m "
        log_string += f"""{title.center(width, " ")}\n"""

    for key, value in metrics.items():
        words = key.split("/")
        if (
            words[0].startswith("episode")
            and "reward" not in words[1]
            and "length" not in words[1]
        ):
            log_string += f"""{f"Mean episode {words[1]}:":>{pad}} {value:.4f}\n"""

    log_string += (
        f"""{"-" * width}\n"""
        f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(time_elapsed))}\n"""
    )
    if "episode/sum_reward" in metrics:
        log_string += (
            f"""{"Mean reward:":>{pad}} {metrics["episode/sum_reward"]:.2f}\n"""
        )
    if "episode/length" in metrics:
        log_string += (
            f"""{"Mean episode length:":>{pad}} {metrics["episode/length"]:.2f}\n"""
        )

    eta = max((time_elapsed / num_steps) * (num_total_steps - num_steps), 0)
    if num_steps > 0 and num_total_steps > 0:
        log_string += (
            f"""{"Computation:":>{pad}} {num_steps / time_elapsed:.0f} steps/s\n"""
            f"""{"ETA:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(eta))}\n"""
            f"""{"Notes:":>{pad}} {notes}\n"""
        )

    print(log_string)


def log_metrics(metrics: Dict[str, Any], num_steps: int, defined_metrics: List[str]):
    """Process and log training metrics to Weights & Biases."""
    grouped_log_data = {"env_step": num_steps}
    for key, value in metrics.items():
        if "reward" in key:
            prefix: str = "Train" if "episode" in key.split("/")[0] else "Evaluate"
            grouped_log_data[f"{prefix}/mean_reward"] = value
        elif "length" in key:
            prefix: str = "Train" if "episode" in key.split("/")[0] else "Evaluate"
            grouped_log_data[f"{prefix}/mean_episode_length"] = value
        elif "loss" in key:
            name = key.split("/")[-1]
            grouped_log_data[f"Loss/{name}"] = value
        elif "sps" in key:
            grouped_log_data["Perf/total_fps"] = value
        elif "eval" in key:
            name = key.split("/")[-1]
            if name.startswith("episode_"):
                name = name.replace("episode_", "")
            grouped_log_data[f"Evaluate/{name}"] = value
        elif "episode" in key:
            grouped_log_data[f"Episode/{key.split('/')[-1]}"] = value

    for key in grouped_log_data:
        if key not in defined_metrics:
            wandb.define_metric(key, step_metric="env_step")
            defined_metrics.append(key)

    wandb.log(grouped_log_data)


def get_body_mass_attr_range(
    robot: Robot,
    body_mass_range: List[float],
    hand_mass_range: List[float],
    other_mass_range: List[float],
    num_envs: int,
):
    """Generates a range of body mass attributes for a robot across multiple environments.

    This function modifies the body mass and inertia of a robot model based on specified
    ranges for different body parts (torso, end-effector, and others) and returns a dictionary
    containing the updated attributes for each environment.

    Args:
        robot (Robot): The robot object containing configuration and name.
        body_mass_range (List[float]): The range of mass deltas for the torso.
        hand_mass_range (List[float]): The range of mass deltas for the end-effector.
        other_mass_range (List[float]): The range of mass deltas for other body parts.
        num_envs (int): The number of environments to generate.

    Returns:
        Dict[str, jax.Array | npt.NDArray[np.float32]]: A dictionary with keys representing
        different body mass attributes and values as JAX arrays or NumPy arrays containing
        the attribute values across all environments.
    """
    xml_path = os.path.join("toddlerbot", "descriptions", robot.name, "scene.xml")
    torso_name = "torso"
    hand_name = robot.config["robot"]["hand_name"]

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    body_mass = model.body_mass.copy()
    body_inertia = model.body_inertia.copy()

    body_mass_delta_list = np.linspace(body_mass_range[0], body_mass_range[1], num_envs)
    hand_mass_delta_list = np.linspace(hand_mass_range[0], hand_mass_range[1], num_envs)
    other_mass_delta_list = np.linspace(
        other_mass_range[0], other_mass_range[1], num_envs
    )
    # Randomize the order of the body mass deltas
    body_mass_delta_list = np.random.permutation(body_mass_delta_list)
    hand_mass_delta_list = np.random.permutation(hand_mass_delta_list)
    other_mass_delta_list = np.random.permutation(other_mass_delta_list)

    # Create lists to store attributes for all environments
    body_mass_list = []
    body_inertia_list = []
    actuator_acc0_list = []
    body_invweight0_list = []
    body_subtreemass_list = []
    dof_M0_list = []
    dof_invweight0_list = []
    tendon_invweight0_list = []
    for body_mass_delta, hand_mass_delta, other_mass_delta in zip(
        body_mass_delta_list, hand_mass_delta_list, other_mass_delta_list
    ):
        # Update body mass and inertia in the model
        for i in range(model.nbody):
            body_name = model.body(i).name

            if body_mass[i] < 1e-6 or body_mass[i] < other_mass_range[1]:
                continue

            if torso_name in body_name:
                mass_delta = body_mass_delta
            elif hand_name in body_name:
                mass_delta = hand_mass_delta
            else:
                mass_delta = other_mass_delta

            model.body(body_name).mass = body_mass[i] + mass_delta
            model.body(body_name).inertia = (
                (body_mass[i] + mass_delta) / body_mass[i] * body_inertia[i]
            )

        mujoco.mj_setConst(model, data)

        # Append the values to corresponding lists
        body_mass_list.append(jnp.array(model.body_mass))
        body_inertia_list.append(jnp.array(model.body_inertia))
        actuator_acc0_list.append(jnp.array(model.actuator_acc0))
        body_invweight0_list.append(jnp.array(model.body_invweight0))
        body_subtreemass_list.append(jnp.array(model.body_subtreemass))
        dof_M0_list.append(jnp.array(model.dof_M0))
        dof_invweight0_list.append(jnp.array(model.dof_invweight0))
        tendon_invweight0_list.append(jnp.array(model.tendon_invweight0))

    # Return a dictionary where each key has a JAX array of all values across environments
    body_mass_attr_range: Dict[str, jax.Array | npt.NDArray[np.float32]] = {
        "body_mass": jnp.stack(body_mass_list),
        "body_inertia": jnp.stack(body_inertia_list),
        "actuator_acc0": jnp.stack(actuator_acc0_list),
        "body_invweight0": jnp.stack(body_invweight0_list),
        "body_subtreemass": jnp.stack(body_subtreemass_list),
        "dof_M0": jnp.stack(dof_M0_list),
        "dof_invweight0": jnp.stack(dof_invweight0_list),
        "tendon_invweight0": jnp.stack(tendon_invweight0_list),
    }

    return body_mass_attr_range


def domain_randomize(
    sys: base.System,
    rng: jax.Array,
    friction_range: List[float],
    damping_range: List[float],
    armature_range: List[float],
    frictionloss_range: List[float],
    body_mass_attr_range: Optional[Dict[str, jax.Array | npt.NDArray[np.float32]]],
) -> Tuple[base.System, base.System]:
    """Randomizes the physical parameters of a system within specified ranges.

    Args:
        sys (base.System): The system whose parameters are to be randomized.
        rng (jax.Array): Random number generator state.
        friction_range (List[float]): Range for randomizing friction values.
        damping_range (List[float]): Range for randomizing damping values.
        armature_range (List[float]): Range for randomizing armature values.
        frictionloss_range (List[float]): Range for randomizing friction loss values.
        body_mass_attr_range (Optional[Dict[str, jax.Array | npt.NDArray[np.float32]]]): Optional dictionary specifying ranges for body mass attributes.

    Returns:
        Tuple[base.System, base.System]: A tuple containing the randomized system and the in_axes configuration for JAX transformations.
    """

    @jax.vmap
    def rand(rng: jax.Array):
        _, rng_friction, rng_damping, rng_armature, rng_frictionloss, rng_qpos0 = (
            jax.random.split(rng, 6)
        )

        friction = jax.random.uniform(
            rng_friction, minval=friction_range[0], maxval=friction_range[1]
        )
        # Two feet and two hands with the floor
        pair_friction = sys.pair_friction.at[:4, :2].set(friction)

        nv = sys.nv - 6

        damping = (
            jax.random.uniform(
                rng_damping,
                (nv,),
                minval=damping_range[0],
                maxval=damping_range[1],
            )
            * sys.dof_damping[6:]
        )
        dof_damping = sys.dof_damping.at[6:].set(damping)

        armature = (
            jax.random.uniform(
                rng_armature,
                (nv,),
                minval=armature_range[0],
                maxval=armature_range[1],
            )
            * sys.dof_armature[6:]
        )
        dof_armature = sys.dof_armature.at[6:].set(armature)

        frictionloss = (
            jax.random.uniform(
                rng_frictionloss,
                (nv,),
                minval=frictionloss_range[0],
                maxval=frictionloss_range[1],
            )
            * sys.dof_frictionloss[6:]
        )
        dof_frictionloss = sys.dof_frictionloss.at[6:].set(frictionloss)

        return pair_friction, dof_damping, dof_armature, dof_frictionloss

    friction, damping, armature, frictionloss = rand(rng)

    body_mass_attr = {}
    if body_mass_attr_range is not None:
        for k, v in body_mass_attr_range.items():
            body_mass_attr[k] = v[: rng.shape[0]]

    in_axes = jax.tree.map(lambda x: None, sys)
    in_axes_dict = {
        "pair_friction": 0,
        "dof_damping": 0,
        "dof_armature": 0,
        "dof_frictionloss": 0,
        **{key: 0 for key in body_mass_attr.keys()},
    }
    in_axes = in_axes.tree_replace(in_axes_dict)

    sys_dict = {
        "pair_friction": friction,
        "dof_damping": damping,
        "dof_armature": armature,
        "dof_frictionloss": frictionloss,
        **body_mass_attr,
    }
    sys = sys.tree_replace(sys_dict)

    return sys, in_axes


def load_runner_config(train_cfg: PPOConfig):
    """Load and configure RSL-RL runner settings from PPO configuration."""
    config_path = os.path.join("toddlerbot", "locomotion", "rsl_rl_config.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

        config["wandb_project"] = train_cfg.wandb_project
        config["wandb_entity"] = train_cfg.wandb_entity
        config["seed"] = train_cfg.seed
        config["max_iterations"] = train_cfg.num_timesteps // (
            train_cfg.num_envs * train_cfg.unroll_length
        )
        config["save_interval"] = config["max_iterations"] // train_cfg.num_evals
        config["render_interval"] = config["max_iterations"] // train_cfg.render_nums
        config["num_steps_per_env"] = train_cfg.unroll_length
        config["empirical_normalization"] = train_cfg.normalize_observation
        config["algorithm"]["gamma"] = train_cfg.discounting
        config["algorithm"]["lam"] = train_cfg.gae_lambda
        config["algorithm"]["max_grad_norm"] = train_cfg.max_grad_norm
        config["algorithm"][
            "normalize_advantage_per_mini_batch"
        ] = not train_cfg.normalize_advantage
        config["algorithm"]["num_learning_epochs"] = train_cfg.num_updates_per_batch
        config["algorithm"]["learning_rate"] = train_cfg.learning_rate
        config["algorithm"]["entropy_coef"] = train_cfg.entropy_cost
        config["algorithm"]["clip_param"] = train_cfg.clipping_epsilon
        config["algorithm"]["num_mini_batches"] = train_cfg.num_minibatches

        config["policy"] = {}
        config["policy"]["actor_hidden_dims"] = train_cfg.policy_hidden_layer_sizes
        config["policy"]["critic_hidden_dims"] = train_cfg.value_hidden_layer_sizes
        config["policy"]["activation"] = train_cfg.activation
        # config["policy"]["distribution_type"] = train_cfg.distribution_type
        config["policy"]["init_noise_std"] = train_cfg.init_noise_std
        config["policy"]["class_name"] = (
            "ActorCriticRecurrent" if train_cfg.use_rnn else "ActorCritic"
        )
        if train_cfg.use_rnn:
            config["policy"]["rnn_type"] = train_cfg.rnn_type
            config["policy"]["rnn_hidden_size"] = train_cfg.rnn_hidden_size
            config["policy"]["rnn_num_layers"] = train_cfg.rnn_num_layers
        else:
            config["policy"]["noise_std_type"] = train_cfg.noise_std_type

    return config


def rollout(jit_reset, jit_step, inference_fn, train_cfg, use_torch, use_batch, rng):
    """Execute policy rollout for evaluation and video generation."""

    def policy(obs):
        """Apply policy to observations and return actions."""
        if use_torch:
            obs_torch = jax_to_torch(obs["state"], device=device)
            if use_batch:
                obs_torch = obs_torch[None].repeat(train_cfg.num_envs, 1)
            action_torch = inference_fn(obs_torch)
            if use_batch:
                action_torch = action_torch[0]
            action = torch_to_jax(action_torch)
        else:
            action = inference_fn(obs, rng)[0]

        return action

    state = jit_reset(rng)
    states = [state]
    # Run one episode for rendering
    for _ in tqdm(range(train_cfg.episode_length), desc="Evaluating"):
        ctrl = policy(state.obs)
        state = jit_step(state, ctrl)
        states.append(state)
        if state.done:
            break

    return states


def train(
    env: MJXEnv,
    eval_env: MJXEnv,
    test_env: MJXEnv,
    train_cfg: PPOConfig,
    run_name: str,
    args: argparse.Namespace,
):
    """Trains a reinforcement learning agent using the Proximal Policy Optimization (PPO) algorithm.

    This function sets up the training environment, initializes configurations, and manages the training process, including saving configurations, logging metrics, and handling checkpoints.

    Args:
        env (MJXEnv): The training environment.
        eval_env (MJXEnv): The evaluation environment.
        make_networks_factory (Any): Factory function to create neural network models.
        train_cfg (PPOConfig): Configuration settings for the PPO training process.
        run_name (str): Name of the training run, used for organizing results.
        restore_path (str): Path to restore a previous checkpoint, if any.
    """
    exp_folder_path = os.path.join("results", run_name)
    ckpt_dir = os.path.join(exp_folder_path, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Store original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Use it like this
    tee_output = Tee(os.path.join(exp_folder_path, "output.log"))
    sys.stdout = tee_output
    sys.stderr = sys.stdout

    restore_params = None
    if len(args.restore) > 0:
        use_torch_model = args.restore.endswith(".pt")
        if args.torch:
            if use_torch_model:
                restore_params = torch.load(args.restore, weights_only=False)
            else:
                jax_params = model.load_params(args.restore)
                restore_params = load_jax_ckpt_to_torch(jax_params)
        else:
            if use_torch_model:
                raise NotImplementedError(
                    "Loading PyTorch model in JAX mode is not implemented yet."
                )
            else:
                restore_params = model.load_params(args.restore)

    args_dict = vars(args)
    with open(os.path.join(exp_folder_path, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=4)

    # print("Arguments:")
    # print(json.dumps(args_dict, indent=4))  # Pretty-print the arguments

    # Save train config to a file and print it
    train_config_dict = dataclass2dict(train_cfg)  # Convert dataclass to dictionary
    with open(os.path.join(exp_folder_path, "train_config.json"), "w") as f:
        json.dump(train_config_dict, f, indent=4)

    # Print the train config
    # print("Train Config:")
    # print(json.dumps(train_config_dict, indent=4))  # Pretty-print the config

    # Save env config to a file and print it
    env_config_dict = dataclass2dict(env.cfg)  # Convert dataclass to dictionary
    with open(os.path.join(exp_folder_path, "env_config.json"), "w") as f:
        json.dump(env_config_dict, f, indent=4)

    # Print the env config
    # print("Env Config:")
    # print(json.dumps(env_config_dict, indent=4))  # Pretty-print the config

    store_code_state(exp_folder_path, ".")

    note_list = []
    if len(args.note) > 0:
        note_list.append(args.note)
    if len(args.gin_config) > 0:
        note_list.extend(args.gin_config.split(","))

    notes = ", ".join(note_list)

    try:
        wandb.init(
            project=train_cfg.wandb_project,
            entity=train_cfg.wandb_entity,
            job_type="train",
            sync_tensorboard=True,
            name=run_name,
            notes=notes,
            config={"args": args_dict, "train": train_config_dict, "env": env_config_dict},
        )
        wandb.define_metric("env_step")
        defined_metrics = ["env_step"]
    except Exception as e:
        print(f"Error with wandb: {e}. \n\nUpdate locomotion/ppo_config.py, wandb_entity with your wandb username, and wandb_project with your project name.")
        defined_metrics = None

    domain_randomize_fn = None
    if env.add_domain_rand:
        body_mass_attr_range = None
        if not env.fixed_base:
            body_mass_attr_range = get_body_mass_attr_range(
                env.robot,
                env.cfg.domain_rand.body_mass_range,
                env.cfg.domain_rand.hand_mass_range,
                env.cfg.domain_rand.other_mass_range,
                train_cfg.num_envs,
            )

        domain_randomize_fn = functools.partial(
            domain_randomize,
            friction_range=env.cfg.domain_rand.friction_range,
            damping_range=env.cfg.domain_rand.damping_range,
            armature_range=env.cfg.domain_rand.armature_range,
            frictionloss_range=env.cfg.domain_rand.frictionloss_range,
            body_mass_attr_range=body_mass_attr_range,
        )

    rng = jax.random.PRNGKey(train_cfg.seed)
    jit_reset = jax.jit(test_env.reset)
    jit_step = jax.jit(test_env.step)

    def render_fn(inference_fn, current_step):
        video_dir = os.path.join(exp_folder_path, "videos")
        states = rollout(
            jit_reset,
            jit_step,
            inference_fn,
            train_cfg,
            args.torch,
            train_cfg.use_rnn,
            rng,
        )
        render_video(test_env, states, video_dir, f"{current_step}")

    times = [time.monotonic()]
    last_ckpt_step = 0
    best_ckpt_step = 0
    best_episode_reward = -float("inf")

    def progress_fn(num_steps: int, metrics: Dict[str, Any]):
        nonlocal defined_metrics, best_episode_reward, best_ckpt_step, last_ckpt_step
        is_episode = any("episode" in k for k in metrics)
        times.append(time.monotonic())
        if is_episode:
            print_metrics(
                metrics, times[-1] - times[0], num_steps, train_cfg.num_timesteps, notes
            )
            if not args.torch:
                last_ckpt_step = num_steps
                episode_reward = float(metrics["episode/sum_reward"])
                if episode_reward > best_episode_reward:
                    best_episode_reward = episode_reward
                    best_ckpt_step = num_steps
        else:
            if not args.torch:
                for key in list(metrics.keys()):
                    if "sps" in key:
                        metrics[key] = num_steps / (times[-1] - times[0])
                    elif "policy" in key:
                        metrics["loss/policy"] = metrics.pop(key)
                    elif "v" in key:
                        metrics["loss/value_func"] = metrics.pop(key)
                    elif "entropy" in key:
                        metrics["loss/entropy"] = metrics.pop(key)
                    elif "total" in key:
                        metrics["loss/total"] = metrics.pop(key)

        # Log metrics to wandb
        if defined_metrics is not None: log_metrics(metrics, num_steps, defined_metrics)

    render_interval = train_cfg.num_timesteps // train_cfg.render_nums
    last_render_step = 0

    def policy_params_fn(current_step: int, make_policy: Any, params: Any):
        nonlocal last_render_step
        policy_path = os.path.join(ckpt_dir, f"model_{current_step}")
        model.save_params(policy_path, params)
        # Handle rendering during training
        if current_step - last_render_step >= render_interval:
            # Create and JIT compile policy and environment functions
            eval_policy = make_policy(params, deterministic=True)
            render_fn(jax.jit(eval_policy), current_step)
            last_render_step = current_step

    try:
        if args.torch:
            # The number of environment steps executed for every training step.
            key = jax.random.PRNGKey(train_cfg.seed)
            _, local_key = jax.random.split(key)
            local_key = jax.random.fold_in(local_key, jax.process_index())
            _, key_env, _ = jax.random.split(local_key, 3)

            v_randomization_fn = None
            if domain_randomize_fn is not None:
                randomization_rng = jax.random.split(key_env, train_cfg.num_envs)
                v_randomization_fn = functools.partial(
                    domain_randomize_fn, rng=randomization_rng
                )

            wrap_for_training = envs.training.wrap

            env = wrap_for_training(
                env,
                episode_length=train_cfg.episode_length,
                randomization_fn=v_randomization_fn,
            )

            rsl_env = RSLRLWrapper(env, device, train_cfg)
            runner_config = load_runner_config(train_cfg)

            if not args.symmetry and "symmetry_cfg" in runner_config["algorithm"]:
                del runner_config["algorithm"]["symmetry_cfg"]

            # Print the env config
            print("Runner Config:")
            print(json.dumps(runner_config, indent=4))  # Pretty-print the config

            states = rsl_env.reset()
            renderer = mujoco.Renderer(env.sys.mj_model, height=480, width=640)
            init_state_dir = os.path.join(exp_folder_path, "init_states")
            os.makedirs(init_state_dir, exist_ok=True)
            for i in tqdm(
                range(0, train_cfg.num_envs, train_cfg.num_envs // 20),
                desc="Rendering initial states",
            ):
                d = mujoco.MjData(env.sys.mj_model)
                d.qpos, d.qvel = states.pipeline_state.q[i], states.pipeline_state.qd[i]
                mujoco.mj_forward(env.sys.mj_model, d)
                renderer.update_scene(d, camera="perspective")
                img = Image.fromarray(renderer.render().astype(np.uint8))
                img.save(os.path.join(init_state_dir, f"env_{i}.png"))

            runner = OnPolicyRunner(
                rsl_env,
                runner_config,
                run_name,
                progress_fn,
                render_fn,
                restore_params,
                device=device,
            )
            runner.learn(runner_config["max_iterations"])

        else:
            make_networks_factory = functools.partial(
                ppo_networks.make_ppo_networks,
                policy_hidden_layer_sizes=train_cfg.policy_hidden_layer_sizes,
                value_hidden_layer_sizes=train_cfg.value_hidden_layer_sizes,
                activation=getattr(linen, train_cfg.activation),
                value_obs_key="privileged_state",
                distribution_type=train_cfg.distribution_type,
                noise_std_type=train_cfg.noise_std_type,
                init_noise_std=train_cfg.init_noise_std,
            )
            ppo.train(
                environment=env,
                num_timesteps=train_cfg.num_timesteps,
                num_envs=train_cfg.num_envs,
                episode_length=train_cfg.episode_length,
                learning_rate=train_cfg.learning_rate,
                entropy_cost=train_cfg.entropy_cost,
                discounting=train_cfg.discounting,
                unroll_length=train_cfg.unroll_length,
                batch_size=train_cfg.batch_size,
                num_minibatches=train_cfg.num_minibatches,
                num_updates_per_batch=train_cfg.num_updates_per_batch,
                normalize_observations=train_cfg.normalize_observation,
                clipping_epsilon=train_cfg.clipping_epsilon,
                gae_lambda=train_cfg.gae_lambda,
                max_grad_norm=train_cfg.max_grad_norm,
                normalize_advantage=train_cfg.normalize_advantage,
                network_factory=make_networks_factory,
                seed=train_cfg.seed,
                num_evals=train_cfg.num_evals,
                eval_env=eval_env,
                randomization_fn=domain_randomize_fn,
                log_training_metrics=True,
                progress_fn=progress_fn,
                policy_params_fn=policy_params_fn,
                restore_params=restore_params,
                run_evals=False,
            )

    except KeyboardInterrupt:
        pass

    finally:
        dump_profiling_data(os.path.join(exp_folder_path, "profile_output.lprof"))

        # Restore original stdout and close the Tee object
        if hasattr(sys.stdout, "close"):
            sys.stdout.close()

        sys.stdout = original_stdout
        sys.stderr = original_stderr

    suffix = ""
    if args.torch:
        best_ckpt_step = runner.best_ckpt
        last_ckpt_step = runner.last_ckpt
        suffix = ".pt"

    available = []
    for name in os.listdir(ckpt_dir):
        if name.startswith("model_"):
            available.append(int(name.split("_")[1].replace(suffix, "")))

    if len(available) > 0:
        closest_best = min(available, key=lambda x: abs(x - best_ckpt_step))
        shutil.copy2(
            os.path.join(ckpt_dir, f"model_{closest_best}{suffix}"),
            os.path.join(exp_folder_path, f"model_best{suffix}"),
        )

        closest_last = min(available, key=lambda x: abs(x - last_ckpt_step))
        shutil.copy2(
            os.path.join(ckpt_dir, f"model_{closest_last}{suffix}"),
            os.path.join(exp_folder_path, f"model_last{suffix}"),
        )
        print(f"Best checkpoint step: {closest_best}")
        print(f"Last checkpoint step: {closest_last}")


def evaluate(
    env: MJXEnv, train_cfg: PPOConfig, policy_path: str, args: argparse.Namespace
):
    """Evaluates a policy in a given environment using a specified network factory and logs the results.

    Args:
        env (MJXEnv): The environment in which the policy is evaluated.
        make_networks_factory (Any): A factory function to create network architectures for the policy.
        run_name (str): The name of the run, used for saving and loading policy parameters.
    """
    rsl_env = RSLRLWrapper(env, device, train_cfg, eval=True)
    runner_config = load_runner_config(train_cfg)
    if not args.symmetry and "symmetry_cfg" in runner_config["algorithm"]:
        del runner_config["algorithm"]["symmetry_cfg"]

    if args.torch:
        # ---- Torch direct export ----
        policy_params = torch.load(policy_path, weights_only=False)
    else:
        jax_params = model.load_params(policy_path)
        policy_params = load_jax_ckpt_to_torch(jax_params)

    runner = OnPolicyRunner(
        rsl_env, runner_config, restore_params=policy_params, device=device
    )

    dummy_input = torch.zeros(
        (1, env.num_obs_history * env.obs_size), dtype=torch.float32
    )
    policy_model = runner.alg.policy.to("cpu")
    policy_model.eval()
    actor_network = policy_model.actor
    try:
        onnx_path = policy_path.replace(".pt", "") + ".onnx"
        torch.onnx.export(
            actor_network,
            dummy_input,
            onnx_path,
            # verbose=True,
            input_names=["obs"],
            output_names=["action"],
        )
        print(f"Policy exported to ONNX at {onnx_path}")
        # Verify the file was actually created
        if os.path.exists(onnx_path):
            print(f"ONNX file size: {os.path.getsize(onnx_path)} bytes")
        else:
            print("ONNX file was not created!")

        if wandb.run is not None:
            artifact = wandb.Artifact(
                name="policy",  # Artifact name
                type="model",  # Artifact type: model, dataset, etc.
                metadata={},
            )
            exp_folder_path = os.path.dirname(policy_path)
            artifact.add_file(onnx_path)
            artifact.add_file(os.path.join(exp_folder_path, "env_config.json"))
            artifact.add_file(os.path.join(exp_folder_path, "train_config.json"))
            artifact.add_file(os.path.join(exp_folder_path, "args.json"))
            wandb.log_artifact(
                artifact, aliases=["latest", os.path.basename(exp_folder_path)]
            )
            print("ONNX artifact logged to wandb.")

    except Exception as e:
        print(f"ONNX export failed: {e}")

    # # for comparison with ONNX
    # torch.save(
    #     actor_network,
    #     os.path.join(os.path.dirname(policy_path), "actor.pt"),
    # )

    # jit_reset = env.reset
    # jit_step = env.step
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    inference_fn = runner.get_inference_policy(device=device)
    # initialize the state
    for seed in [0, 1, 2, 3]:
        rng = jax.random.PRNGKey(seed)
        states = rollout(jit_reset, jit_step, inference_fn, train_cfg, True, False, rng)
        video_name = f"eval_{seed}"
        video_path = os.path.join(os.path.dirname(policy_path), f"{video_name}.mp4")
        render_video(
            env,
            states,
            os.path.dirname(policy_path),
            video_name,
            cameras=["perspective", "side", "top", "front"],
        )
        if wandb.run is not None:
            wandb.log({"video": wandb.Video(video_path, format="mp4")})


def main(args=None):
    """Trains or evaluates a policy for a specified robot and environment using PPO.

    This function sets up the training or evaluation of a policy for a robot in a specified environment. It parses command-line arguments to configure the robot, environment, evaluation settings, and other parameters. It then loads configuration files, binds any overridden parameters, and initializes the environment and robot. Depending on the arguments, it either trains a new policy or evaluates an existing one.

    Args:
        args (list, optional): List of command-line arguments. If None, arguments are parsed from sys.argv.

    Raises:
        FileNotFoundError: If a specified gin configuration file or evaluation run is not found.
    """
    parser = argparse.ArgumentParser(description="Train the mjx policy.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot_2xc",
        help="The name of the robot. Need to match the name in descriptions.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="walk",
        help="The name of the env.",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default="",
        help="Provide the time string of the run to evaluate.",
    )
    parser.add_argument(
        "--restore",
        type=str,
        default="",
        help="Path to the checkpoint folder.",
    )
    parser.add_argument(
        "--ref",
        type=str,
        default="",
        help="Name of the reference motion.",
    )
    parser.add_argument(
        "--gin-file",
        type=str,
        default="",
        help="List of gin config files",
    )
    parser.add_argument(
        "--gin-config",
        type=str,
        default="",
        help="Override gin config parameters (e.g., SimConfig.timestep=0.01 ObsConfig.frame_stack=10)",
    )
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="Add notes to wandb.",
    )
    parser.add_argument(
        "--brax",
        action="store_true",
        default=False,
        help="Use brax instead of torch.",
    )
    parser.add_argument(
        "--symmetry",
        action="store_true",
        default=False,
        help="Use symmetry in rsl_rl.",
    )

    args = parser.parse_args()
    args.torch = not args.brax

    gin_file_list = [args.env] + args.gin_file.split(" ")
    for gin_file in gin_file_list:
        if len(gin_file) == 0:
            continue

        gin_file_path = os.path.join(
            os.path.dirname(__file__),
            gin_file + ".gin" if not gin_file.endswith(".gin") else gin_file,
        )
        if not os.path.exists(gin_file_path):
            raise FileNotFoundError(f"File {gin_file_path} not found.")

        gin.parse_config_file(gin_file_path)

    # Bind parameters from --gin_config
    if len(args.gin_config) > 0:
        overrides = [s.strip() for s in args.gin_config.split(";") if s.strip()]
        gin.parse_config(overrides)

    robot = Robot(args.robot)

    EnvClass = get_env_class(args.env.replace("_fixed", ""))
    env_cfg = MJXConfig()
    train_cfg = PPOConfig()

    assert train_cfg.num_envs == train_cfg.num_minibatches * train_cfg.batch_size, (
        "The number of environments must match the number of minibatches times the batch size."
    )

    kwargs = {}
    if len(args.ref) > 0:
        kwargs = {"ref_motion_type": args.ref}

    env = EnvClass(
        args.env,
        robot,
        env_cfg,  # type: ignore
        fixed_base="fixed" in args.env,
        add_domain_rand=env_cfg.domain_rand.add_domain_rand,
        **kwargs,  # type: ignore
    )
    eval_env = EnvClass(
        args.env,
        robot,
        env_cfg,  # type: ignore
        fixed_base="fixed" in args.env,
        add_domain_rand=env_cfg.domain_rand.add_domain_rand,
        **kwargs,  # type: ignore
    )
    test_env = EnvClass(
        args.env,
        robot,
        env_cfg,  # type: ignore
        fixed_base="fixed" in args.env,
        add_domain_rand=env_cfg.domain_rand.add_domain_rand,
        **kwargs,
    )
    test_env.rand_init_state_indices = [0]

    if len(args.eval) > 0:
        if os.path.exists(args.eval):
            evaluate(test_env, train_cfg, args.eval, args)
        else:
            raise FileNotFoundError(f"{args.eval} not found.")
    else:
        if len(args.eval) > 0:
            time_str = args.eval
        else:
            time_str = time.strftime("%Y%m%d_%H%M%S")

        codebase = "rsl" if args.torch else "brax"
        run_name = f"{robot.name}_{args.env}_{codebase}_{time_str}"

        train(env, eval_env, test_env, train_cfg, run_name, args)

        suffix = ".pt" if args.torch else ""
        policy_path = os.path.join("results", run_name, f"model_best{suffix}")
        if not os.path.exists(policy_path):
            policy_path = os.path.join("results", run_name, f"model_last{suffix}")

        evaluate(test_env, train_cfg, policy_path, args)


if __name__ == "__main__":
    main()
