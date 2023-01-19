######################################################################
# ML/MM: https://github.com/emedio/embedding
#
# Copyright: 2022-2023
#
# Authors: Lester Hedges   <lester.hedges@gmail.com>
#          Kirill Zinovjev <kzinovjev@gmail.com>
#
# ML/MM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# ML/MM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ML/MM. If not, see <http://www.gnu.org/licenses/>.
######################################################################

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

        print(f"Connecting socket at address: ({host}, {port})")
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

        print(f"Binding socket to address: ({host}, {port})")
        self._sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))

    def listen(self, num_connections=1):
        """
        Listen for the specified number of connections. If num_connections = 0,
        then the system default will be used.

        Parameters
        ----------

        num_connections : int
        """
        if type(num_connections) is not int:
            raise ValueError("'num_connections' must be of type 'int'")
        if num_connections < 0:
            raise ValueError("'num_connections' must be >= 0")

        self._sock.listen(num_connections)

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
