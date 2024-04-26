"""
    Use scp to send file to a node with ip and dst path.
    Note: send own ip will cause warning, ignore it
"""
import os

class Dispatcher:
    def __init__(self, ip_table):
        self.ip_table = ip_table
        self.ip_dict = {}
        self.init_ip_dict()

    def init_ip_dict(self, ):
        # node ip keys
        for ip in self.ip_table:
            if ip in self.ip_dict: continue
            self.ip_dict[ip] = True

    def renew_ip_table(self, ip_table):
        self.ip_table = ip_table
        self.init_ip_dict()

    def dispatch_ip_table(self, src_file, dst_path):
        """
            master node send file to each node
        """
        for ip in self.ip_table:
            os.system("scp %s %s:%s" % (src_file, ip, dst_path))
        print("ip table dispatched.")
    
    # each local rank 0 dispatch to each other node
    def dispatch_detected_topo(self, src_file, dst_path):
        """
            only send file to local rank 0 on each node
        """        
        for ip in self.ip_dict.keys():
            os.system("scp %s %s:%s" % (src_file, ip, dst_path))
        print("detected topology dispatched.")

    # each local rank 0 sends to world rank 0
    def send_profiled_topo(self, src_file, dst_path):
        """
            send file to world rank node
        """
        os.system("scp %s %s:%s" % (src_file, self.ip_table[0], dst_path))
        print("profiled topology sent to master.")

    def dispatch_strategy(self, src_file, dst_path):
        """
            master send file to each node
        """
        for ip in self.ip_dict.keys():
            os.system("scp %s %s:%s" % (src_file, ip, dst_path))
        print("strategy dispatched.")
