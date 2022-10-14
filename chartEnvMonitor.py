import time
import json
from logging import getLogger


from gym.utils import atomic_write
from gym.utils.json_utils import json_encode_np
from gym.wrappers import Monitor as _GymMonitor
from gym.wrappers.monitoring.stats_recorder import StatsRecorder as _GymStatsRecorder


class Monitor(_GymMonitor):
	"""`Monitor` with PFRL's `ContinuingTimeLimit` support.

	`Agent` in PFRL might reset the env even when `done=False`
	if `ContinuingTimeLimit` returns `info['needs_reset']=True`,
	which is not expected for `gym.Monitor`.

	For details, see
	https://github.com/openai/gym/blob/master/gym/wrappers/monitor.py
	"""

	def _start(
		self,
		directory,
		video_callable=None,
		force=False,
		resume=False,
		write_upon_reset=False,
		uid=None,
		mode=None,
	):
		if self.env_semantics_autoreset:
			raise NotImplementedError(
				"Detect 'semantics.autoreset=True' in `env.metadata`, "
				"which means the env is from deprecated OpenAI Universe."
			)
		ret = super()._start(
			directory=directory,
			video_callable=video_callable,
			force=force,
			resume=resume,
			write_upon_reset=write_upon_reset,
			uid=uid,
			mode=mode,
		)
		env_id = self.stats_recorder.env_id
		self.stats_recorder = _StatsRecorder(
			directory,
			"{}.episode_batch.{}".format(self.file_prefix, self.file_infix),
			vega_dataset=self.env.get_data(),
			autoreset=False,
			env_id=env_id,
		)
		if mode is not None:
			self._set_mode(mode)
		return ret

	def step(self, action):
		self._before_step(action)
		observation, reward, done, info = self.env.step(action)
		states = self.env.return_wrapped_config()
		history = self.env.history
		done = self._after_step(observation, reward, done, (states, history), info)

		return observation, reward, done, info

	def _after_step(self, observation, reward, done, res, info):
		if not self.enabled:
			return done

		if done and self.env_semantics_autoreset:
			# For envs with BlockingReset wrapping VNCEnv, this observation will be the first one of the new episode
			self.reset_video_recorder()
			self.episode_id += 1
			self._flush()

		# Record stats
		self.stats_recorder.after_step(observation, reward, done, *res, info)
		# Record video
		self.video_recorder.capture_frame()

		return done


class _StatsRecorder(_GymStatsRecorder):
	"""`StatsRecorder` with PFRL's `ContinuingTimeLimit` support.

	For details, see
	https://github.com/openai/gym/blob/master/gym/wrappers/monitoring/stats_recorder.py
	"""

	def __init__(
		self,
		directory,
		file_prefix,
		vega_dataset,
		autoreset=False,
		env_id=None,
		logger=getLogger(__name__),
	):
		super().__init__(directory, file_prefix, autoreset=autoreset, env_id=env_id)
		self.vega_dataset = vega_dataset
		self.episode_dashboard_states = []
		self.episode_dashboard_hists = []
		self.dashboard_states = []
		self._save_completed = True
		self.logger = logger

	def before_reset(self):
		assert not self.closed

		if self.done is not None and not self.done and self.steps > 0:
			self.logger.debug(
				"Tried to reset the env which is not done=True. "
				"StatsRecorder completes the last episode."
			)
			self.save_complete()

		self.done = False
		if self.initial_reset_timestamp is None:
			self.initial_reset_timestamp = time.time()

	def after_step(self, observation, reward, done, dashboard_states, dashboard_hist, info):
		self._save_completed = False
		self.dashboard_states = dashboard_states
		self.dashboard_hist = dashboard_hist
		self.steps += 1
		self.total_steps += 1
		self.rewards += reward
		self.done = done

		if done:
			self.save_complete()

		if done:
			if self.autoreset:
				self.before_reset()
				self.after_reset(observation)

	def save_complete(self):
		if not self._save_completed:
			super().save_complete()
			self._save_completed = True

	def save_complete(self):
		if not self._save_completed:
			if self.steps is not None:
				self.episode_lengths.append(self.steps)
				self.episode_rewards.append(float(self.rewards))
				self.timestamps.append(time.perf_counter())
				self.episode_dashboard_states.append(self.dashboard_states)
				self.episode_dashboard_hists.append(self.dashboard_hist)
			self._save_completed = True

	def flush(self):
		if self.closed:
			return

		with atomic_write.atomic_write(self.path) as f:
			json.dump(
				{
					"initial_reset_timestamp": self.initial_reset_timestamp,
					"timestamps": self.timestamps,
					"episode_lengths": self.episode_lengths,
					"episode_rewards": self.episode_rewards,
					"episode_types": self.episode_types,
					"episode_dashboard_states": self.episode_dashboard_states,
					"episode_dashboard_hists": self.episode_dashboard_hists,
					"dataset": self.vega_dataset
				},
				f,
				default=json_encode_np,
			)

	def close(self):
		self.save_complete()
		super().close()
