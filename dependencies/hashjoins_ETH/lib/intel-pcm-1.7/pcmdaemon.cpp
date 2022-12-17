/*
Copyright (c) 2011, Intel Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
//
// contact: Roman Dementiev
//

#include <fcntl.h>
#include <sys/stat.h>
#include <mqueue.h>
#include <iostream>
#include <errno.h>

// This is work in progress. The file is not used right now.

#define PCM_REQ_QUEUE_NAME "/IntelPCM_request_queue"

int main()
{
    // TODO: set umask
    //umask(0777);
    mqd_t req_queue = mq_open(PCM_REQ_QUEUE_NAME, O_EXCL | O_CREAT | O_RDONLY, S_IRWXU | S_IRWXG | S_IRWXO, NULL);

    if (req_queue == ((mqd_t)-1))
    {
        std::cerr << "Can not open IntelPCM request queue. Error: " << errno << std::endl;
        return -1;
    }

    char buffer[16 * 1024];

    ssize_t msg_size = mq_receive(req_queue, buffer, 16 * 1024, 0);

    if (msg_size == -1)
    {
        std::cout << "Error while receiving message. Error: " << errno << std::endl;
        mq_unlink(PCM_REQ_QUEUE_NAME);
        return -1;
    }

    std::cout << "Received message" << buffer << std::endl;

    mq_unlink(PCM_REQ_QUEUE_NAME);

    return 0;
}
