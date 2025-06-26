import os
import shutil
from pathlib import Path

def reorganize_replica_dataset(root_dir):
    root_path = Path(root_dir)
    
    # 遍历每个房间（如room0, room1等）
    for room_dir in root_path.iterdir():
        if not room_dir.is_dir():
            continue
            
        print(f"Processing {room_dir.name}...")
        
        # 原始路径
        results_dir = room_dir / "results"
        if not results_dir.exists():
            print(f"Warning: 'results' directory not found in {room_dir}")
            continue
        
        # 创建新目录
        images_dir = room_dir / "images"
        depth_dir = room_dir / "depth_images"
        images_dir.mkdir(exist_ok=True)
        depth_dir.mkdir(exist_ok=True)
        
        # 移动文件
        for file in results_dir.iterdir():
            if "frame" in file.name and file.suffix in (".jpg", ".png"):
                shutil.move(str(file), str(images_dir / file.name))
            elif "depth" in file.name and file.suffix in (".jpg", ".png"):
                shutil.move(str(file), str(depth_dir / file.name))
        
        # 移动轨迹文件（如果存在）
        traj_file = room_dir / "traj.txt"
        if traj_file.exists():
            # 如果已经在正确位置就不移动
            pass
        else:
            # 检查是否在results目录中
            src_traj = results_dir / "traj.txt"
            if src_traj.exists():
                shutil.move(str(src_traj), str(traj_file))
        
        # 删除空的results目录
        try:
            results_dir.rmdir()
        except OSError:
            print(f"Warning: Could not remove {results_dir} - may not be empty")
        
        print(f"Finished processing {room_dir.name}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", help="Path to the Replica dataset root directory")
    args = parser.parse_args()
    
    reorganize_replica_dataset(args.root_dir)
