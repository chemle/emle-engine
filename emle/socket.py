#######################################################################
# EMLE-Engine: https://github.com/chemle/emle-engine
#
# Copyright: 2023
#
# Authors: Lester Hedges   <lester.hedges@gmail.com>
#          Kirill Zinovjev <kzinovjev@gmail.com>
#
# EMLE-Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# EMLE-Engine is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EMLE-Engine If not, see <http://www.gnu.org/licenses/>.
#####################################################################

import socket as _socket

__all__ = ["Socket"]


class Socket:
    """
    A wrapper around socket.socket with convenience functions to aid
    the sending and receiving of messages.
    """

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

        print(f"Listening for connections from {num_connections} clients...")
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

    def send(self, msg, length):
        """
        Send a message over the socket. The message is ":" token delimited
        string where the header contains the length of the subsequent message.

        For example, for the client:

          "23:emlerun:/path/to/client"

        This communicates an EMLE calculation request to the server, also
        passing the path from which the client was run.

        On succesful completion, the server will return a fixed message of:

          "7:emlefin"

        Parameters
        ----------

        msg : str
            The message to send.

        length : int
            The length of the message.
        """
        if not isinstance(msg, str):
            raise TypeError("'msg' must be of type 'str'")

        if not isinstance(length, int):
            raise TypeError("'length' must be of type 'int'")

        # Encode the message.
        msg_bytes = msg.encode("utf-8")

        totalsent = 0
        while totalsent < length:
            sent = self._sock.send(msg_bytes[totalsent:])
            if sent == 0:
                raise RuntimeError("The socket connection was broken!")
            totalsent = totalsent + sent

    def receive(self):
        """
        Receive a message.

        Returns
        -------

        msg : str
            The message received.

        path/error : str (optional)
            The path from which the message was sent on the client, or the
            error message returned from the server.
        """

        # First work out the message length.
        buf = b""
        while True:
            c = self._sock.recv(1)
            if not c:
                print("The socket connection was broken!")
                return None, None
            if c == b":":
                break
            else:
                buf += c

        # Store the length of the message.
        msg_len = int(buf)

        chunks = []
        bytes_recd = 0

        # Now read the rest of the message.
        while bytes_recd < msg_len:
            chunk = self._sock.recv(min(msg_len - bytes_recd, msg_len))
            if chunk == b"":
                print("The socket connection was broken!")
                return None, None
            chunks.append(chunk)
            bytes_recd = bytes_recd + len(chunk)

        data = b"".join(chunks).decode("utf-8").split(":")

        if len(data) < 2:
            return data[0], None
        else:
            return data[:2]
