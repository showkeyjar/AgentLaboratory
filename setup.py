from config import ConfigManager
import os

def setup_project():
    """项目配置向导"""
    print("\n欢迎使用项目配置向导！\n")
    config_manager = ConfigManager()
    
    # 生成配置文件
    env_file = "config.env"
    if os.path.exists(env_file):
        print("检测到已存在的配置文件，是否要重新配置？")
        print("1. 是")
        print("2. 否")
        choice = input("请输入选择 (1/2): ")
        if choice != "1":
            print("\n配置已取消。")
            return
    
    with open(env_file, "w", encoding="utf-8") as f:
        for config_name, config_data in config_manager.configs.items():
            print(f"\n{config_data['question']}")
            
            # 显示选项
            for key, option in config_data['options'].items():
                print(f"{key}. {option}")
            
            # 获取用户选择
            while True:
                choice = input("\n请输入选择: ")
                if choice in config_data['options']:
                    break
                print("无效的选择，请重新输入！")
            
            # 写入配置
            f.write(f"{config_name}={choice}\n")
            
    print("\n配置完成！配置文件已保存为 config.env")
    print("请根据需要修改 config.env 中的具体值")

if __name__ == '__main__':
    setup_project()
