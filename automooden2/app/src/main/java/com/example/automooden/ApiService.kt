package com.example.automooden

import retrofit2.http.GET
import retrofit2.http.Path

interface ApiService {

    @GET("/api/car/{plate}")
    suspend fun getCar(

        @Path("plate")
        plate: String

    ): CarResponse
}