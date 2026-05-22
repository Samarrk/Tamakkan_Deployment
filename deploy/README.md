# Jetson deployment

## Install systemd service

    sudo cp deploy/tamakkan.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable tamakkan
    sudo systemctl start tamakkan

## Common operations

- Live logs: journalctl -u tamakkan -f
- Status: sudo systemctl status tamakkan
- Restart: sudo systemctl restart tamakkan

## Health check from another host

    curl http://<jetson-ip>:8000/health

Expect: {"status":"ok","pipeline_loaded":true,"active_session_id":null}
