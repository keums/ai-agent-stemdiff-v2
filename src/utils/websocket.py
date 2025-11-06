# utils/task_notification.py
import json
import boto3
import logging
import os
import requests
import time
from typing import Any, Dict, Optional
from .cache import cache_get, cache_set, cache_delete

logger = logging.getLogger()

def get_emulator_url():
    """Get the QCW emulator base URL"""
    return os.environ.get('QCW_EMULATOR_URL', 'http://localhost:5000')

def is_local_development():
    """Check if running in local development environment"""
    return os.environ.get('ENVIRONMENT', '').lower() == 'local'

def send_websocket_message(connection_id: str, message: Dict[str, Any]) -> bool:
    try:
        websocket_api_endpoint = os.environ.get('WEBSOCKET_API_ENDPOINT').replace('wss://', 'https://')  # Assuming the endpoint is stored in the function's environment variables
        apigateway_management_api = boto3.client('apigatewaymanagementapi', endpoint_url=websocket_api_endpoint)

        apigateway_management_api.post_to_connection(
            Data=json.dumps(message),
            ConnectionId=connection_id
        )
        return True
    except Exception as e:
        logger.error(f"Error sending WebSocket message: {str(e)}")
        if "GoneException" in str(e):
            # Connection is no longer valid
            return False
        raise

def store_websocket_connection(task_id: str, connection_id: str, api_id: str, stage: str) -> bool:
    """
    Store WebSocket connection information
    
    Args:
        task_id (str): The task ID
        connection_id (str): The WebSocket connection ID
        api_id (str): The API Gateway ID
        stage (str): The API stage
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if is_local_development():
            response = requests.post(
                f"{get_emulator_url()}/api/websocket/connection",
                json={
                    'task_id': task_id,
                    'connection_id': connection_id,
                    'api_id': api_id,
                    'stage': stage
                }
            )
            return response.status_code == 201
        else:
            websocket_data = {
                'connection_id': connection_id,
                'api_id': api_id,
                'stage': stage
            }
            return cache_set(f"websocket:{task_id}", websocket_data)
    except Exception as e:
        logger.error(f"Error storing WebSocket connection: {str(e)}")
        return False

def get_websocket_connection(task_id: str) -> Optional[Dict[str, str]]:
    """
    Get stored WebSocket connection information
    
    Args:
        task_id (str): The task ID
        
    Returns:
        Optional[Dict[str, str]]: The WebSocket connection data or None if not found
    """
    try:
        if is_local_development():
            response = requests.get(f"{get_emulator_url()}/api/websocket")
            if response.status_code == 200:
                data = response.json()
                for conn in data['connections']:
                    if conn['task_id'] == task_id:
                        return {
                            'connection_id': conn['connection_id'],
                            'api_id': conn['api_id'],
                            'stage': conn['stage']
                        }
            return None
        else:
            return cache_get(f"websocket:{task_id}")
    except Exception as e:
        logger.error(f"Error getting WebSocket connection: {str(e)}")
        return None

def remove_websocket_connection(task_id: str) -> bool:
    """
    Remove stored WebSocket connection information
    
    Args:
        task_id (str): The task ID
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if is_local_development():
            # For the emulator, we'll update the connection with a null connection_id
            response = requests.post(
                f"{get_emulator_url()}/api/websocket/connection",
                json={
                    'task_id': task_id,
                    'connection_id': None,
                    'api_id': None,
                    'stage': None
                }
            )
            return response.status_code == 201
        else:
            return cache_delete(f"websocket:{task_id}")
    except Exception as e:
        logger.error(f"Error removing WebSocket connection: {str(e)}")
        return False

def update_task_status(task_id: str, status: str, message: str, progress: float, data: Optional[Dict[str, Any]] = None) -> bool:
    """
    Update task status and send notification via WebSocket if connected
    
    Args:
        task_id (str): The task ID
        status (str): The new status
        message (str): The status message
        progress (float): The progress value (0-100)
        data (Optional[Dict[str, Any]]): Additional data to include
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if is_local_development():
            # Update task status
            response = requests.post(
                f"{get_emulator_url()}/api/websocket/task",
                json={
                    'task_id': task_id,
                    'status': status,
                    'message': message,
                    'progress': progress,
                    'data': data
                }
            )
            if response.status_code != 201:
                return False

            # Add log entry
            response = requests.post(
                f"{get_emulator_url()}/api/websocket/log",
                json={
                    'task_id': task_id,
                    'message': message,
                    'type': 'status'
                }
            )
            if response.status_code != 201:
                return False

            # Get connection info and send WebSocket message if connected
            websocket_data = get_websocket_connection(task_id)
            if websocket_data:
                task_data = {
                    'status': status,
                    'message': message,
                    'progress': progress,
                    'data': data,
                    'last_updated': time.time()
                }
                return send_websocket_message(
                    websocket_data['connection_id'],
                    websocket_data['api_id'],
                    websocket_data['stage'],
                    task_data
                )
            return True
        else:
            # Get current task state
            task = cache_get(f"task:{task_id}")
            if not task:
                logger.warning(f"Cannot update task {task_id}: Task not found")
                return False
            
            # Update task data
            task.update({
                'status': status,
                'message': message,
                'progress': progress,
                'data': data,
                'last_updated': time.time()
            })
            
            # Add log entry
            if 'logs' not in task:
                task['logs'] = []
            task['logs'].append({'message': message, 'type': 'status'})
            
            # Save updated task
            if not cache_set(f"task:{task_id}", task):
                return False
            
            # Check if WebSocket connection exists
            websocket_data = get_websocket_connection(task_id)
            if websocket_data:
                try:
                    return send_websocket_message(
                        websocket_data['connection_id'],
                        websocket_data['api_id'],
                        websocket_data['stage'],
                        task
                    )
                except Exception as e:
                    logger.error(f"Error sending WebSocket update: {str(e)}")
                    if "GoneException" in str(e):
                        remove_websocket_connection(task_id)
                        logger.info(f"Removed stale WebSocket connection for task {task_id}")
            
            return False
    except Exception as e:
        logger.error(f"Error updating task status: {str(e)}")
        return False