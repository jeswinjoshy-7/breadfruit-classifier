def application(environ, start_response):
    """WSGI application entry point for Vercel Python framework"""
    start_response('200 OK', [('Content-Type', 'text/plain')])
    return [b'Hello from Vercel Python framework']