from neo4j import GraphDatabase
import socket

# 1. 首先检查端口是否开放 (更底层的检查)
def check_port(host='localhost', port=7687):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2) # 2秒超时
        try:
            s.connect((host, port))
            return True
        except:
            return False

print("1. 正在检查 Neo4j 默认端口 (7687)...")
if check_port():
    print("✅ 端口 7687 开放，大概率有 Neo4j 正在运行！")
else:
    print("❌ 端口 7687 未开放，Neo4j 未启动或未安装。")
    print("   如果你安装了 Neo4j Desktop，请确保在软件里点击了 'Start' 启动数据库。")
    exit()

# 2. 尝试使用密码登录 (请修改为你记得的密码，默认账号通常是 neo4j，初始密码可能是 neo4j 或 password)
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "password" # ⚠️ 这里改成你可能用过的密码

print(f"\n2. 正在尝试使用账号 {USER} 登录数据库...")
try:
    # 尝试建立连接
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    # 验证连接是否有效
    driver.verify_connectivity()
    print("🎉 恭喜！成功连接到 Neo4j 数据库，环境完美就绪！")
    driver.close()
except Exception as e:
    print("❌ 账号密码错误，或连接被拒绝。详细报错如下：")
    print(e)