import os
import h5py

import mimicgen.utils.file_utils as MG_FileUtils


def test_parse_source_dataset():
    # test with a mock source dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "tmp", "tmp.hdf5")
    demo_keys = ["demo_0", "demo_1"]

    num_subtasks = 4
    subtask_term_signals = [
        f"subtask_{i_subtask+1}" for i_subtask in range(num_subtasks - 1)
    ]
    subtask_term_signals.append(None)
    subtask_term_offset_ranges = [(0, 0) for _ in range(num_subtasks - 1)]
    subtask_term_offset_ranges.append(None)  # last subtask is None

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    with h5py.File(dataset_path, "w") as f:
        f_data = f.create_group("data")
        demo_len = num_subtasks + 1
        for demo_key in demo_keys:
            f_demo = f_data.create_group(demo_key)

            # dummy action data
            f_demo.create_dataset("actions", data=[0.0] * demo_len)

            f_datagen = f_demo.create_group("datagen_info")

            # dummy datagen info
            f_datagen.create_dataset("eef_pose", data=[0.0] * demo_len)
            f_datagen.create_dataset("target_pose", data=[0.0] * demo_len)
            f_datagen.create_dataset("gripper_action", data=[0.0] * demo_len)

            # dummy object poses
            dummy_objects = ["object_1", "object_2"]
            f_datagen.create_group("object_poses")
            for obj in dummy_objects:
                f_datagen["object_poses"].create_dataset(obj, data=[0.0] * demo_len)

            # dummy subtask signals
            if demo_key == demo_keys[0]:
                # all tasks are completed once
                # {[0,1,1,1,1], [0,0,1,1,1], [0,0,0,1,1], [0,0,0,0,1]}
                f_datagen.create_group("subtask_term_signals")
                for i_subtask, subtask_key in enumerate(subtask_term_signals):
                    if subtask_key is None:
                        subtask_key = f"subtask_{i_subtask+1}"
                    f_datagen["subtask_term_signals"].create_dataset(
                        subtask_key,
                        data=[int(ii > i_subtask) for ii in range(demo_len)],
                    )
            elif demo_key == demo_keys[1]:
                # subtask_3 is completed twice
                # {[0,1,1,1,1], [0,0,1,1,1], [0,1,0,1,1], [0,0,0,0,1]}
                f_datagen.create_group("subtask_term_signals")
                for i_subtask, subtask_key in enumerate(subtask_term_signals):
                    if subtask_key is None:
                        subtask_key = f"subtask_{i_subtask+1}"
                    if subtask_key == "subtask_3":
                        # subtask_3 is completed twice
                        f_datagen["subtask_term_signals"].create_dataset(
                            subtask_key,
                            data=[0, 1, 0, 1, 1],
                        )
                    else:
                        f_datagen["subtask_term_signals"].create_dataset(
                            subtask_key,
                            data=[int(ii > i_subtask) for ii in range(demo_len)],
                        )

    try:
        _ = MG_FileUtils.parse_source_dataset(
            dataset_path,
            demo_keys,
            subtask_term_signals=subtask_term_signals,
            subtask_term_offset_ranges=subtask_term_offset_ranges,
        )
    except Exception as e:
        raise e
    finally:
        # remove temporary dataset file
        os.remove(dataset_path)
        os.rmdir(os.path.dirname(dataset_path))
