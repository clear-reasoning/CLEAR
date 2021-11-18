import boto3
import json
from urllib.parse import unquote_plus
from run.data_pipeline import AthenaQuery

queryEngine = AthenaQuery()
sqs = boto3.client('sqs')
succeeded = []

def lambda_handler(event, context):
    for record in event['Records']:
        message_body = json.loads(record['body'])
        try:
            readied_query_name, result_location, query_date, partition, inflow_filter, outflow_filter, start_filter, max_decel, leader_max_decel= message_body
            queryEngine.run_query(readied_query_name, result_location, query_date, partition, inflow_filter=inflow_filter, outflow_filter=outflow_filter,
                                  start_filter=start_filter, max_decel=max_decel, leader_max_decel=leader_max_decel)
        except Exception as e:
            # handle partial batch failure by manually deleting successful messages
            for receipt_handle in succeeded:
                response = sqs.delete_message(
                                QueueUrl="https://sqs.us-west-2.amazonaws.com/409746595792/RunQueryRequests",
                                ReceiptHandle=receipt_handle
                              )
            raise
        succeeded.append(record["receiptHandle"])