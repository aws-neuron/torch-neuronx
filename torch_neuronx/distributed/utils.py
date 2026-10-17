import socket


def get_free_port():
    """Return an available ephemeral TCP port number.

    Returns:
        str: Port number selected by the operating system.
    """
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return str(port)
