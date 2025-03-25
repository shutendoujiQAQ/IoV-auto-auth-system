#!/usr/bin/env python3
import socket
import threading
import json
import argparse
import time
import os
import base64

class Node:
    def __init__(self, node_id, rank, host, port, auth_flag, send_dir, recv_dir, peers_file, data_flag_file, blacklist_file, discovery_port):
        """
        初始化节点参数：
          - node_id: 节点唯一标识
          - rank: 用于选举的整数（数值越大优先）
          - host, port: 当前节点的监听地址
          - auth_flag: 认证标志（0或1），只有值为1时才能加入区块链网络
          - send_dir: 待发送文件所在目录
          - recv_dir: 接收文件存储目录
          - peers_file: 单独的peers列表文件路径（JSON格式），用于动态更新网络成员信息
          - data_flag_file: 数据发送标志文件路径，内容为 "0" 或 "1"
          - blacklist_file: 黑名单列表文件路径（JSON格式），保存被禁止的节点ID列表
          - discovery_port: UDP广播用于链发现的端口
        """
        self.node_id = node_id
        self.rank = rank
        self.host = host
        self.port = port
        self.auth_flag = int(auth_flag)
        self.send_dir = send_dir
        self.recv_dir = recv_dir
        self.peers_file = peers_file
        self.data_flag_file = data_flag_file
        self.blacklist_file = blacklist_file
        self.discovery_port = discovery_port

        # PBFT相关变量
        self.current_view = 0           # 当前视图号
        self.current_primary = None     # 当前主节点标识
        self.sequence_num = 1           # 交易序列号，从1开始
        self.prepare_messages = {}      # {sequence: set(节点ID)}
        self.commit_messages = {}       # {sequence: set(节点ID)}
        self.lock = threading.Lock()
        self.is_primary = False         # 是否为当前主节点

        # 动态成员管理
        self.peers = []                 # 从peers_file获取的当前网络成员（字典列表）
        self.blacklist = set()          # 从blacklist_file读取的独立黑名单集合
        self.reconfig_version = 0       # 重配置版本

        # 心跳检测
        self.last_hb_time = time.time()

    def start(self):
        """启动节点前先进行认证和链发现，认证通过后启动各模块线程"""
        if self.auth_flag != 1:
            print(f"节点 {self.node_id} 认证未通过，无法加入区块链网络。")
            return
        print(f"节点 {self.node_id} 认证通过，启动节点服务。")

        # 启动UDP发现服务器线程（用于响应链发现请求）
        threading.Thread(target=self.start_discovery_server, args=(self.discovery_port,), daemon=True).start()
        time.sleep(1)

        # 新节点通过UDP广播进行链发现
        responses = self.chain_discovery(self.discovery_port)
        if responses:
            # 收到响应，说明已有链存在，新节点更新自己的peers列表
            with self.lock:
                # 此处简单将所有响应作为peers列表（后续也会通过文件动态更新）
                self.peers = responses
            print(f"节点 {self.node_id} 发现已有链，peers信息：{self.peers}")
        else:
            print(f"节点 {self.node_id} 未发现可加入的链，将自己作为第一个节点。")

        # 启动TCP服务器线程
        threading.Thread(target=self.start_server, daemon=True).start()
        time.sleep(1)

        # 启动动态更新peers和黑名单的线程（每5秒检查一次）
        threading.Thread(target=self.update_peers, daemon=True).start()
        # 等待初次更新后再确定主节点
        time.sleep(2)
        self.update_primary()

        # 启动心跳发送线程，每秒发送一次HEARTBEAT
        threading.Thread(target=self.heartbeat_sender, daemon=True).start()
        # 启动检测主节点状态线程
        threading.Thread(target=self.check_primary, daemon=True).start()
        # 启动数据发送检测线程
        threading.Thread(target=self.check_data_flag, daemon=True).start()

        # 如果当前节点为主节点，则延时后模拟发起共识
        if self.is_primary:
            time.sleep(2)
            self.initiate_consensus("Dummy Request Data")

    def start_discovery_server(self, discovery_port):
        """
        启动UDP发现服务器，监听来自其他新节点的链发现请求，
        并回复自己的节点信息（作为链中已有成员）。
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('', discovery_port))
        print(f"节点 {self.node_id} UDP发现服务器启动，监听端口 {discovery_port}")
        while True:
            try:
                data, addr = s.recvfrom(4096)
                message = json.loads(data.decode())
                if message.get("type") == "CHAIN_DISCOVERY_REQUEST":
                    response = {
                        "type": "CHAIN_DISCOVERY_RESPONSE",
                        "node_id": self.node_id,
                        "host": self.host,
                        "port": self.port,
                        "rank": self.rank,
                        "trust": 1  # 此处可扩展为实际信任值
                    }
                    s.sendto(json.dumps(response).encode(), addr)
                    print(f"节点 {self.node_id} 响应来自 {addr} 的链发现请求")
            except Exception as e:
                print(f"节点 {self.node_id} UDP发现服务器异常：{e}")

    def chain_discovery(self, discovery_port):
        """
        新节点通过UDP广播发送链发现请求，等待响应收集已有链中成员信息，
        超时后返回响应列表（可能为空）。
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.settimeout(5)  # 设置超时时间为5秒
        request = {
            "type": "CHAIN_DISCOVERY_REQUEST",
            "node_id": self.node_id
        }
        broadcast_address = ('<broadcast>', discovery_port)
        try:
            s.sendto(json.dumps(request).encode(), broadcast_address)
            print(f"节点 {self.node_id} 发送链发现广播请求")
        except Exception as e:
            print(f"节点 {self.node_id} 发送链发现广播失败：{e}")
        responses = []
        start_time = time.time()
        while time.time() - start_time < 5:
            try:
                data, addr = s.recvfrom(4096)
                resp = json.loads(data.decode())
                if resp.get("type") == "CHAIN_DISCOVERY_RESPONSE":
                    responses.append(resp)
                    print(f"节点 {self.node_id} 收到链发现响应：{resp}")
            except socket.timeout:
                break
            except Exception as e:
                print(f"节点 {self.node_id} 接收链发现响应错误：{e}")
        s.close()
        return responses

    def update_primary(self):
        """
        根据当前视图和所有节点的rank信息（排除黑名单中的节点），确定当前主节点。
        规则：将本节点与peers（已过滤黑名单）按rank降序、node_id升序排序，
        然后取索引为 current_view mod 总节点数 的节点作为主节点。
        """
        with self.lock:
            all_nodes = [(self.node_id, self.rank)]
            for peer in self.peers:
                all_nodes.append((peer['node_id'], peer['rank']))
            all_nodes.sort(key=lambda x: (-x[1], x[0]))
            if len(all_nodes) == 0:
                return
            index = self.current_view % len(all_nodes)
            new_primary = all_nodes[index][0]
            self.current_primary = new_primary
            if self.node_id == new_primary:
                self.is_primary = True
                print(f"节点 {self.node_id} 成为主节点, 视图 {self.current_view}")
            else:
                self.is_primary = False
                print(f"节点 {self.node_id} 的当前主节点为 {new_primary}, 视图 {self.current_view}")

    def start_server(self):
        """启动TCP服务器，监听其他节点发送的消息"""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((self.host, self.port))
        s.listen(5)
        print(f"节点 {self.node_id} 正在 {self.host}:{self.port} 上监听")
        while True:
            conn, addr = s.accept()
            threading.Thread(target=self.handle_connection, args=(conn,), daemon=True).start()

    def handle_connection(self, conn):
        """处理每个入站连接，接收数据后解析消息；若发送者在黑名单中则忽略"""
        with conn:
            data = conn.recv(4096)
            if not data:
                return
            try:
                message = json.loads(data.decode())
                sender = message.get("sender")
                if sender in self.blacklist:
                    print(f"节点 {self.node_id} 忽略来自黑名单节点 {sender} 的消息")
                    return
                self.process_message(message)
            except Exception as e:
                print(f"节点 {self.node_id} 解析消息出错: {e}")

    def send_message(self, peer_host, peer_port, message):
        """向指定peer发送消息，采用TCP连接"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((peer_host, peer_port))
            s.sendall(json.dumps(message).encode())
            s.close()
        except Exception as e:
            print(f"节点 {self.node_id} 向 {peer_host}:{peer_port} 发送消息失败 - {e}")

    def broadcast_message(self, message):
        """向当前peers列表中所有节点广播消息（已过滤黑名单）"""
        with self.lock:
            for peer in self.peers:
                self.send_message(peer['host'], peer['port'], message)

    def initiate_consensus(self, request_data):
        """
        主节点发起共识流程：构造并广播PRE_PREPARE消息，
        并在本地处理该消息。
        """
        pre_prepare_msg = {
            "type": "PRE_PREPARE",
            "view": self.current_view,
            "sequence": self.sequence_num,
            "payload": request_data,
            "sender": self.node_id,
            "signature": "simulated_signature"
        }
        print(f"主节点 {self.node_id} 广播 PRE_PREPARE 消息，序列号 {self.sequence_num}")
        self.broadcast_message(pre_prepare_msg)
        self.process_message(pre_prepare_msg)

    def process_message(self, message):
        """根据消息类型调用相应处理函数"""
        msg_type = message.get("type")
        if msg_type == "PRE_PREPARE":
            self.handle_pre_prepare(message)
        elif msg_type == "PREPARE":
            self.handle_prepare(message)
        elif msg_type == "COMMIT":
            self.handle_commit(message)
        elif msg_type == "HEARTBEAT":
            self.handle_heartbeat(message)
        elif msg_type == "VIEW_CHANGE":
            self.handle_view_change(message)
        elif msg_type == "FILE_TRANSFER":
            self.handle_file_transfer(message)
        elif msg_type == "RECONFIG":
            self.handle_reconfig(message)
        # 可扩展其他消息类型

    def handle_pre_prepare(self, message):
        """处理PRE_PREPARE消息，并广播PREPARE消息"""
        seq = message["sequence"]
        print(f"节点 {self.node_id} 收到来自 {message['sender']} 的 PRE_PREPARE 消息，序列号 {seq}")
        prepare_msg = {
            "type": "PREPARE",
            "view": message["view"],
            "sequence": seq,
            "payload": message["payload"],
            "sender": self.node_id,
            "signature": "simulated_signature"
        }
        self.broadcast_message(prepare_msg)
        self.process_message(prepare_msg)

    def handle_prepare(self, message):
        """处理PREPARE消息，统计达到阈值后广播COMMIT消息"""
        seq = message["sequence"]
        with self.lock:
            if seq not in self.prepare_messages:
                self.prepare_messages[seq] = set()
            self.prepare_messages[seq].add(message["sender"])
            count = len(self.prepare_messages[seq])
        print(f"节点 {self.node_id} 收到 PREPARE 消息数（序列号 {seq}）：{count}")
        threshold = 2 * ((len(self.peers) + 1 - 1) // 3) + 1  # 计算2f+1（此处f根据总节点数计算）
        if count >= threshold and seq not in self.commit_messages:
            commit_msg = {
                "type": "COMMIT",
                "view": message["view"],
                "sequence": seq,
                "payload": message["payload"],
                "sender": self.node_id,
                "signature": "simulated_signature"
            }
            self.broadcast_message(commit_msg)
            self.process_message(commit_msg)

    def handle_commit(self, message):
        """处理COMMIT消息，统计达到阈值后认为共识达成并回复客户端（打印执行结果）"""
        seq = message["sequence"]
        with self.lock:
            if seq not in self.commit_messages:
                self.commit_messages[seq] = set()
            self.commit_messages[seq].add(message["sender"])
            count = len(self.commit_messages[seq])
        print(f"节点 {self.node_id} 收到 COMMIT 消息数（序列号 {seq}）：{count}")
        threshold = 2 * ((len(self.peers) + 1 - 1) // 3) + 1
        if count >= threshold:
            print(f"节点 {self.node_id} 对序列号 {seq} 达成共识，提交请求：{message['payload']}")
            reply_msg = {
                "type": "REPLY",
                "view": message["view"],
                "sequence": seq,
                "payload": f"Executed {message['payload']}",
                "sender": self.node_id,
                "signature": "simulated_signature"
            }
            print(f"节点 {self.node_id} 回复客户端：{reply_msg['payload']}")

    def heartbeat_sender(self):
        """每隔1秒广播一次HEARTBEAT消息"""
        while True:
            hb_msg = {
                "type": "HEARTBEAT",
                "view": self.current_view,
                "sender": self.node_id,
                "timestamp": time.time()
            }
            self.broadcast_message(hb_msg)
            self.process_message(hb_msg)
            time.sleep(1)

    def check_primary(self):
        """
        每秒检测当前主节点是否正常：
        若当前节点非主节点且距离上次收到主节点心跳超过3秒，则发起视图转换。
        """
        timeout = 3
        while True:
            if not self.is_primary:
                elapsed = time.time() - self.last_hb_time
                if elapsed > timeout:
                    print(f"节点 {self.node_id} 检测到主节点 {self.current_primary} 心跳超时（{elapsed:.1f}s），发起视图转换")
                    new_view = self.current_view + 1
                    self.initiate_view_change(new_view)
            time.sleep(1)

    def initiate_view_change(self, new_view):
        """
        发起视图转换：广播VIEW_CHANGE消息，并直接更新视图号。
        """
        vc_msg = {
            "type": "VIEW_CHANGE",
            "view": new_view,
            "sender": self.node_id,
            "signature": "simulated_signature"
        }
        self.broadcast_message(vc_msg)
        self.handle_view_change(vc_msg)

    def handle_view_change(self, message):
        """
        处理VIEW_CHANGE消息：
        若收到的视图号高于当前视图，则更新视图、重新计算主节点，并重置心跳计时。
        """
        new_view = message.get("view")
        with self.lock:
            if new_view > self.current_view:
                print(f"节点 {self.node_id} 更新视图，从 {self.current_view} 到 {new_view}")
                self.current_view = new_view
                self.update_primary()
                self.last_hb_time = time.time()

    def handle_heartbeat(self, message):
        """处理HEARTBEAT消息：若来自当前主节点，则更新last_hb_time"""
        sender = message.get("sender")
        if sender == self.current_primary:
            self.last_hb_time = time.time()

    def check_data_flag(self):
        """
        定时检测data_flag_file内容，
        若内容为"1"，则读取send_dir下所有文件并发送给网络中所有节点，
        发送完毕后将data_flag_file内容重置为"0"。
        """
        while True:
            try:
                with open(self.data_flag_file, "r") as f:
                    flag = f.read().strip()
                if flag == "1":
                    print(f"节点 {self.node_id} 检测到数据发送标志，开始发送文件")
                    self.send_files()
                    with open(self.data_flag_file, "w") as f:
                        f.write("0")
            except Exception as e:
                print(f"节点 {self.node_id} 检查data_flag_file出错: {e}")
            time.sleep(1)

    def send_files(self):
        """读取send_dir下所有文件，将内容（经过base64编码）封装后以FILE_TRANSFER消息广播"""
        if not os.path.isdir(self.send_dir):
            print(f"发送目录 {self.send_dir} 不存在")
            return
        for filename in os.listdir(self.send_dir):
            file_path = os.path.join(self.send_dir, filename)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "rb") as f:
                        file_data = f.read()
                    encoded_data = base64.b64encode(file_data).decode()
                    file_msg = {
                        "type": "FILE_TRANSFER",
                        "filename": filename,
                        "content": encoded_data,
                        "sender": self.node_id,
                        "timestamp": time.time()
                    }
                    self.broadcast_message(file_msg)
                    print(f"节点 {self.node_id} 发送文件 {filename}")
                except Exception as e:
                    print(f"节点 {self.node_id} 读取文件 {filename} 出错: {e}")

    def handle_file_transfer(self, message):
        """处理FILE_TRANSFER消息，将接收到的文件保存到recv_dir下"""
        filename = message.get("filename")
        content = message.get("content")
        if not filename or not content:
            print(f"节点 {self.node_id} 收到非法文件传输消息")
            return
        try:
            file_data = base64.b64decode(content)
            if not os.path.isdir(self.recv_dir):
                os.makedirs(self.recv_dir)
            file_path = os.path.join(self.recv_dir, filename)
            with open(file_path, "wb") as f:
                f.write(file_data)
            print(f"节点 {self.node_id} 成功接收并保存文件 {filename}")
        except Exception as e:
            print(f"节点 {self.node_id} 保存文件 {filename} 出错: {e}")

    def update_peers(self):
        """
        每5秒从peers_file中读取最新的网络成员（JSON格式），
        同时从blacklist_file中读取独立黑名单，
        对比新成员列表与当前本地成员列表（以节点ID比较），
        若检测到变化则发起重配置。
        """
        while True:
            try:
                new_peers = []
                if os.path.exists(self.peers_file):
                    with open(self.peers_file, "r") as f:
                        peers_list = json.load(f)
                    for peer in peers_list:
                        if int(peer.get("trust", 1)) > 0:
                            new_peers.append(peer)
                        else:
                            print(f"节点 {self.node_id} 检测到节点 {peer['node_id']} trust<=0，跳过该节点")
                new_blacklist = set()
                if os.path.exists(self.blacklist_file):
                    with open(self.blacklist_file, "r") as f:
                        blacklist_data = json.load(f)
                    new_blacklist = set(blacklist_data)
                with self.lock:
                    self.blacklist = new_blacklist
                    current_ids = sorted([p["node_id"] for p in self.peers])
                    new_ids = sorted([p["node_id"] for p in new_peers])
                if current_ids != new_ids:
                    print(f"节点 {self.node_id} 检测到成员变化，新成员: {new_ids}, 当前成员: {current_ids}")
                    self.initiate_reconfiguration(new_peers)
            except Exception as e:
                print(f"节点 {self.node_id} 更新peers出错: {e}")
            time.sleep(5)

    def initiate_reconfiguration(self, new_members):
        """
        发起重配置（RECONFIG）：构造并广播RECONFIG消息，
        消息中包含新的成员列表和一个递增的重配置版本号。
        """
        with self.lock:
            new_version = self.reconfig_version + 1
        reconfig_msg = {
            "type": "RECONFIG",
            "reconfig_version": new_version,
            "new_members": new_members,
            "sender": self.node_id,
            "timestamp": time.time(),
            "signature": "simulated_signature"
        }
        print(f"节点 {self.node_id} 发起重配置，版本 {new_version}")
        self.broadcast_message(reconfig_msg)
        self.handle_reconfig(reconfig_msg)

    def handle_reconfig(self, message):
        """
        处理RECONFIG消息：若消息中的重配置版本大于本地版本，
        则更新本地的成员列表、重配置版本，并重新计算主节点。
        """
        msg_version = message.get("reconfig_version")
        with self.lock:
            if msg_version > self.reconfig_version:
                self.reconfig_version = msg_version
                self.peers = message.get("new_members", [])
                print(f"节点 {self.node_id} 更新成员列表，重配置版本 {self.reconfig_version}")
                self.update_primary()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node_id", required=True, help="节点的唯一标识")
    parser.add_argument("--rank", type=int, required=True, help="用于选举的整数值，数值越大优先")
    parser.add_argument("--host", default="localhost", help="绑定的主机地址")
    parser.add_argument("--port", type=int, required=True, help="监听端口")
    parser.add_argument("--auth_flag", required=True, help="认证标志，必须为1才能加入网络")
    parser.add_argument("--send_dir", required=True, help="待发送文件所在目录")
    parser.add_argument("--recv_dir", required=True, help="接收文件存储目录")
    parser.add_argument("--peers_file", required=True, help="动态peers列表文件路径（JSON格式）")
    parser.add_argument("--data_flag_file", required=True, help="数据发送标志文件路径，内容为0或1")
    parser.add_argument("--blacklist_file", required=True, help="黑名单列表文件路径（JSON格式）")
    parser.add_argument("--discovery_port", type=int, default=50000, help="UDP链发现使用的端口")
    args = parser.parse_args()

    node = Node(
        node_id=args.node_id,
        rank=args.rank,
        host=args.host,
        port=args.port,
        auth_flag=args.auth_flag,
        send_dir=args.send_dir,
        recv_dir=args.recv_dir,
        peers_file=args.peers_file,
        data_flag_file=args.data_flag_file,
        blacklist_file=args.blacklist_file,
        discovery_port=args.discovery_port
    )
    node.start()

    # 主线程保持运行
    while True:
        time.sleep(1)
