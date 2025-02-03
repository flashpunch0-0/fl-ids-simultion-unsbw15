import flwr as fl
import os
from web3 import Web3

# Blockchain Configuration
web3 = Web3(Web3.HTTPProvider(INFURA_URL))
# INFURA_URL = "https://sepolia.infura.io/v3/de3a5bd9bb4e4044aa516db5facf2dcc"

# PRIVATE_KEY = "42de94928c34120381ea51a905d7af4df84e46189735b203032ba1c7bd58f059"
# ACCOUNT_ADDRESS = "0x03eDf8C9f29E6C6DdA629a62A8674801F877d109"
# CONTRACT_ADDRESS = "0xaB3963B038045b8b6095565B3EeB8C46F316Aaf8"


PRIVATE_KEY = os.getenv("PRIVATE_KEY")
INFURA_URL = os.getenv("INFURA_URL")
ACCOUNT_ADDRESS = os.getenv("ACCOUNT_ADDRESS")
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS")


CONTRACT_ABI = [
	{
		"anonymous": False,
		"inputs": [
			{
				"indexed": True,
				"internalType": "uint256",
				"name": "round",
				"type": "uint256"
			},
			{
				"indexed": False,
				"internalType": "uint256",
				"name": "accuracy",
				"type": "uint256"
			},
			{
				"indexed": False,
				"internalType": "uint256",
				"name": "loss",
				"type": "uint256"
			}
		],
		"name": "MetricsUpdated",
		"type": "event"
	},
	{
		"inputs": [
			{
				"internalType": "uint256",
				"name": "round",
				"type": "uint256"
			},
			{
				"internalType": "uint256",
				"name": "accuracy",
				"type": "uint256"
			},
			{
				"internalType": "uint256",
				"name": "loss",
				"type": "uint256"
			}
		],
		"name": "storeMetrics",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "uint256",
				"name": "round",
				"type": "uint256"
			}
		],
		"name": "getMetrics",
		"outputs": [
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			},
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			}
		],
		"name": "globalMetrics",
		"outputs": [
			{
				"internalType": "uint256",
				"name": "accuracy",
				"type": "uint256"
			},
			{
				"internalType": "uint256",
				"name": "loss",
				"type": "uint256"
			}
		],
		"stateMutability": "view",
		"type": "function"
	}
]

contract = web3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)
# Make tensorflow log less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def send_to_blockchain(round_num, metrics):
    """Send aggregated metrics to Ethereum blockchain."""
    accuracy = int(metrics.get("accuracy", 0) * 1000)  # Convert to integer (fixed-point representation)
    loss = int(metrics.get("loss", 0) * 1000)

    gas_estimate = contract.functions.storeMetrics(round_num, accuracy, loss).estimate_gas({
        "from": ACCOUNT_ADDRESS
    })
    nonce = web3.eth.get_transaction_count(ACCOUNT_ADDRESS)
    tx = contract.functions.storeMetrics(round_num, accuracy, loss).build_transaction({
    "from": ACCOUNT_ADDRESS,
    "gas": gas_estimate,  # Use estimated gas
    "gasPrice": web3.to_wei("5", "gwei"),
    "nonce": nonce,
})


    signed_tx = web3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    print(f"Sent aggregated metrics to blockchain: {web3.to_hex(tx_hash)}")

# 
def weighted_average(metrics):
    total_examples = 0
    federated_metrics = {k: 0 for k in metrics[0][1].keys()}
    for num_examples, m in metrics:
        for k, v in m.items():
            federated_metrics[k] += num_examples * v
        total_examples += num_examples
    return {k: v / total_examples for k, v in federated_metrics.items()}

def get_server_strategy():
    return fl.server.strategy.FedAvg(
            min_fit_clients=3,
            min_evaluate_clients = 3,
            min_available_clients=3,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    
if __name__ == "__main__":
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=get_server_strategy(),
        config=fl.server.ServerConfig(num_rounds=3),
    )
    final_round, acc = history.metrics_distributed["accuracy"][-1]
    print(f"After {final_round} rounds of training the accuracy is {acc:.3%}")
    send_to_blockchain(final_round, history.metrics_distributed)