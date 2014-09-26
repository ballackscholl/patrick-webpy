# -*- coding:utf-8 -*-
'''
Created on 2014/9/25

@author: yu.zhou
'''
from web import config
from web.session import Store
import logging

class RedisStore(Store):

    def __init__(self, **kwargs):
        from redis import Redis

        host = 'localhost'
        if 'host' in kwargs:
            host = kwargs['host']

        port = 6379
        if 'port' in kwargs:
            port = kwargs['port']

        password = None
        if 'password' in kwargs:
            password = kwargs['password']

        encoding = 'utf-8'
        if 'encoding' in kwargs:
            encoding = kwargs['encoding']

        db = 0
        if 'db' in kwargs:
            db = kwargs['db']

        self.server = Redis(host=host, port=port,  db=db, password=password, socket_keepalive= True, encoding=encoding)

    def __contains__(self, key):
        if key in self.server:
            return True
        return False

    def __getitem__(self, key):
        pickled = self.server.get(key)
        if pickled:
            return self.decode(pickled)

        raise KeyError, key

    def __setitem__(self, key, value):
        pickled = self.encode(value)
        try:
            self.server.setex(key, pickled, config.session_parameters['timeout'])
        except IOError,e:
            logging.error(str(e))

    def __delitem__(self, key):
        self.server.delete(key)

    def cleanup(self, timeout):
        pass
