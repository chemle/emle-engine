import socket as _socket

__all__ = ["Socket"]


class Socket:
    """
    A wrapper around socket.socket with convenience functions to aid
    the sending and receiving of messages.
    """

    # The length of messages passed via this socket.
    msg_len = 7

    def __init__(self, sock=None):
        """
        Constructor.

        Parameters
        ----------

        sock : socket.socket
            An existing socket object.
        """
        if sock is None:
            self._sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        else:
            if not isinstance(sock, _socket.socket):
                raise TypeError("'sock' must be of type 'socket.sock'")
            else:
                self._sock = sock

    def connect(self, host, port):
        """
        Connect to a specified host and port.

        Parameters
        ----------

        host : str
            The hostname.

        port : int
            The port number.
        """
        if not isinstance(host, str):
            raise TypeError("'host' must be of type 'str'")

        if type(port) is not int:
            raise TypeError("'port' must be of type 'int'")

        print(f"Connecting socket to {host} port {port}")
        self._sock.connect((host, port))

    def bind(self, host, port):
        """
        Bind to a specified host and port.

        Parameters
        ----------

        host : str
            The hostname.

        port : int
            The port number.
        """
        if not isinstance(host, str):
            raise TypeError("'host' must be of type 'str'")

        if type(port) is not int:
            raise TypeError("'port' must be of type 'int'")

        print(f"Binding socket to {host} port {port}")
        self._sock.bind((host, port))

    def listen(self):
        """Enable server to accept a single connection."""
        self._sock.listen(1)

    def accept(self):
        """
        Accept a conection.

        Returns
        -------

        connection : socket.socket
            A new socket to be used for sending and receiving data on
            the connection.

        address : (str, str)
            The address bound to the socket on the other end of the
            connection.
        """
        connection, address = self._sock.accept()
        return Socket(connection), address

    def close(self):
        """Close the socket."""
        self._sock.close()

    def send(self, msg):
        """
        Send a message over the socket.

        Parameters
        ----------

        msg : str
            The message to send.
        """
        if not isinstance(msg, str):
            raise TypeError("'msg' must be of type 'str'")

        # Encode the message.
        msg_bytes = msg.encode("utf-8")

        totalsent = 0
        while totalsent < self.msg_len:
            sent = self._sock.send(msg_bytes[totalsent:])
            if sent == 0:
                raise RuntimeError("The socket connection was broken")
            totalsent = totalsent + sent

    def receive(self):
        chunks = []
        bytes_recd = 0
        while bytes_recd < self.msg_len:
            chunk = self._sock.recv(min(self.msg_len - bytes_recd, 2048))
            if chunk == b"":
                print("The socket connection was broken")
                return None
            chunks.append(chunk.decode("utf-8"))
            bytes_recd = bytes_recd + len(chunk)
        return "".join(chunks)
